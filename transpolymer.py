import torch, sys, math, os, json, copy, random
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel
from transpolymer_tokenizer import PolymerSmilesTokenizer
from transpolymer_dataset import Downstream_Dataset
from copy import deepcopy


class DownstreamRegression(nn.Module):
    def __init__(self, dropout, hidden_dropout, attention_dropout):
        super(DownstreamRegression, self).__init__()
        PretrainedModel = RobertaModel.from_pretrained('data/transpolymer_pretrain.pt')
        tokenizer = PolymerSmilesTokenizer.from_pretrained('roberta-base', max_len=411)
        PretrainedModel.config.hidden_dropout_prob = hidden_dropout
        PretrainedModel.config.attention_probs_dropout_prob = attention_dropout
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        
        self.Regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output


def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)
    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)
        lr *= 0.9

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)
    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    return AdamW(opt_parameters, lr=lr)


def train(bs, lr, lr_reg, LLRD, df_train, df_test, best=False):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)

    y_train = df_train['tg'].to_numpy()
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    kf = KFold(n_splits=5)
    min_model_all = []
    train_loss_all = []
    val_loss_all = []
    val_RMSE = 0

    for i, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(df_train)))):

        model = DownstreamRegression(0.1, 0.1, 0.1).to('cuda')
        model = model.double()
        criterion = nn.MSELoss()

        if LLRD == 1:
            optimizer = roberta_base_AdamW_LLRD(model, lr, 1e-2)
        else:
            optimizer = AdamW(
                [
                    {"params": model.PretrainedModel.parameters(), "lr": lr,
                        "weight_decay": 0.0},
                    {"params": model.Regressor.parameters(), "lr": lr_reg,
                        "weight_decay": 1e-2},
                ]
            )

        train_set = Downstream_Dataset(df_train.loc[train_idx, :].reset_index(drop=True), tokenizer, 411)
        val_set = Downstream_Dataset(df_train.loc[val_idx, :].reset_index(drop=True), tokenizer, 411)
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=bs)

        steps_per_epoch = len(train_loader)
        training_steps = int(steps_per_epoch * 30)
        warmup_steps = int(training_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps)

        torch.cuda.empty_cache()

        min_val_loss = math.inf
        min_model = model.state_dict()
        train_loss = []
        val_loss = []

        for _ in range(30):
            model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to('cuda')
                attention_mask = batch["attention_mask"].to('cuda')
                y_true = batch["prop"].to('cuda').float()
                y_true = (y_true - y_mean) / y_std
                optimizer.zero_grad()
                y_pred = model(input_ids, attention_mask).float()
                loss = criterion(y_pred.squeeze(), y_true.squeeze())
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train_loader)
            train_loss.append(epoch_train_loss)

            epoch_val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to('cuda')
                    attention_mask = batch["attention_mask"].to('cuda')
                    y_true = batch["prop"].to('cuda').float()
                    y_true = (y_true - y_mean) / y_std
                    y_pred = model(input_ids, attention_mask).float()
                    loss = criterion(y_pred.squeeze(), y_true.squeeze())
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)
            print(epoch_val_loss)
            sys.stdout.flush()
            val_loss.append(epoch_val_loss)

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                min_model = copy.deepcopy(model.state_dict())
        
        if best:
            min_model_all.append(min_model)
            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)
            
        model.load_state_dict(min_model, strict=True)

        tg_true = []
        tg_pred = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to('cuda')
                attention_mask = batch["attention_mask"].to('cuda')
                y_true = batch["prop"].to('cuda').float()
                y_true = (y_true - y_mean) / y_std
                y_pred = model(input_ids, attention_mask).float()
                if len(y_pred) == 1:
                    tg_true.append((y_true * y_std + y_mean).item())
                    tg_pred.append((y_pred * y_std + y_mean).item())
                else:
                    tg_true += (y_true * y_std + y_mean).tolist()
                    tg_pred += (y_pred * y_std + y_mean).tolist()
        val_RMSE += root_mean_squared_error(tg_true, tg_pred)

    if best:
        pd.DataFrame({'train_loss1': train_loss_all[0], 'val_loss1': val_loss_all[0], 
                      'train_loss2': train_loss_all[1], 'val_loss2': val_loss_all[1], 
                      'train_loss3': train_loss_all[2], 'val_loss3': val_loss_all[2], 
                      'train_loss4': train_loss_all[3], 'val_loss4': val_loss_all[3], 
                      'train_loss5': train_loss_all[4], 'val_loss5': val_loss_all[4]}).to_csv('results/transpolymer/loss.txt', index=False, float_format='%.4f')
        
        for i in range(5):
            torch.save(min_model_all[i], 'results/transpolymer/model%d.pt' % (i + 1))

        test_set = Downstream_Dataset(df_test, tokenizer, 411)
        test_loader = DataLoader(test_set, batch_size=bs)

        R2 = []
        MAE = []
        MAPE = []
        RMSE = []
        tg_pred_all = []

        for i in range(5):
            model = DownstreamRegression(0.1, 0.1, 0.1).to('cuda')
            model = model.double()
            model.load_state_dict(min_model_all[i], strict=True)
            model.eval()

            tg_true = []
            tg_pred = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to('cuda')
                    attention_mask = batch["attention_mask"].to('cuda')
                    y_true = batch["prop"].to('cuda').float()
                    y_true = (y_true - y_mean) / y_std
                    y_pred = model(input_ids, attention_mask).float().view(-1)
                    if len(y_pred) == 1:
                        tg_true.append((y_true * y_std + y_mean).item())
                        tg_pred.append((y_pred * y_std + y_mean).item())
                    else:
                        tg_true += (y_true * y_std + y_mean).tolist()
                        tg_pred += (y_pred * y_std + y_mean).tolist()

            R2.append(r2_score(tg_true, tg_pred))
            MAE.append(mean_absolute_error(tg_true, tg_pred))
            MAPE.append(mean_absolute_percentage_error(tg_true, tg_pred))
            RMSE.append(root_mean_squared_error(tg_true, tg_pred))
            tg_pred_all.append(tg_pred)

        pd.DataFrame({'true': tg_true,
              'pred1': tg_pred_all[0],
              'pred2': tg_pred_all[1], 
              'pred3': tg_pred_all[2],
              'pred4': tg_pred_all[3],
              'pred5': tg_pred_all[4]}).to_csv('results/transpolymer/pred.txt', index=False, float_format='%.4f')
        pd.DataFrame({'R2': R2, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}).to_csv('results/transpolymer/metrics.txt', index=False, float_format='%.4f')
    return val_RMSE / 5


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

tokenizer = PolymerSmilesTokenizer.from_pretrained('roberta-base', max_len=411)

df_train = pd.read_csv('data/labeled_train.csv')
df_test = pd.read_csv('data/labeled_test.csv')

param = []
rmse_all = []
bs_all = [32, 32, 32, 32, 64, 64, 64, 64, 32, 32, 64, 64]
lr_all = [1e-4, 1e-4, 5e-5, 5e-5, 1e-4, 1e-4, 5e-5, 5e-5, 1e-4, 5e-5, 1e-4, 5e-5]
lr_reg_all = [1e-4, 5e-5, 1e-4, 5e-5, 1e-4, 5e-5, 1e-4, 5e-5, 0, 0, 0, 0]
LLRD_all = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
for i in range(12):
    random.seed(i)
    bs = bs_all[i]
    lr = lr_all[i]
    lr_reg = lr_reg_all[i]
    LLRD = LLRD_all[i]
    RMSE = train(bs, lr, lr_reg, LLRD, df_train, df_test)
    print(bs, lr, lr_reg, LLRD, RMSE)
    sys.stdout.flush()

    param.append({'batch size': bs,
                  'learning rate': lr,
                  'learning rate regressor': lr,
                  'LLRD': LLRD})
    rmse_all.append(RMSE)

idx_min = np.argmin(np.array(rmse_all))
print('Best RMSE: %.4f' % (rmse_all[idx_min]))

if not os.path.exists('results/transpolymer'):
    os.mkdir('results/transpolymer')

train(param[idx_min]['batch size'],
      param[idx_min]['learning rate'],
      param[idx_min]['learning rate regressor'],
      param[idx_min]['LLRD'],
      df_train, df_test, True)

with open('results/transpolymer/hyperparameters.txt', 'w') as file:
     file.write(json.dumps(param[idx_min]))
