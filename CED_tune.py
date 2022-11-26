import numpy as np
import json
import pandas as pd
from IPython.display import display
import argparse
from datetime import timedelta

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import seed_everything
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from torchmetrics.functional import confusion_matrix

from ray import tune
from ray.tune import PlacementGroupFactory
from ray.tune.schedulers import ASHAScheduler



seed_everything(42, workers=True)

max_len = 100
K = 30
length = 30
vocabulary_size = 20000
#Add unknown and pad tokens
vocabulary_size += 2
embedding_size = 200
num_checkpoints = 5
# the threshold value
alpha = 0.95
# hyper-params in AttenCED loss
lambd0 = 0.01
lambd1 = 0.2

base_path = '' # base path
checkpoint_base_path = '' # path to save checkpoints

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.rnn_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor, lens):

        X = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        output, h_n = self.rnn_encoder(X)
        h_n = torch.reshape(h_n, (h_n.size()[1], h_n.size()[0], h_n.size()[2]))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, h_n


class CEDCNN(pl.LightningModule):

    def __init__(
            self,
            lambda0,
            lambda1,
            num_hidden=100,
            num_filters=50,
            num_classes=2,
            filter_sizes=(4, 5),
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.num_hidden = num_hidden
        self.num_filters = num_filters
        self.num_filters_total = num_filters * len(filter_sizes)
        self.num_classes = num_classes
        self.rnn = RNNEncoder(self.num_filters_total, self.num_hidden)
        self.linear = nn.Linear(
            num_hidden + (len(filter_sizes) * num_filters), self.num_classes)
        self.conv_om_1 = nn.Sequential(
            nn.Conv2d(1, self.num_filters,
                      (filter_sizes[0], embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((length - filter_sizes[0] + 1, 1), 1)
        )
        self.conv_om_2 = nn.Sequential(
            nn.Conv2d(1, self.num_filters,
                      (filter_sizes[1], embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((length - filter_sizes[1] + 1, 1), 1)
        )
        self.conv_post_1 = nn.Sequential(
            nn.Conv2d(1, self.num_filters,
                      (filter_sizes[0], embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((K - filter_sizes[0] + 1, 1), 1)
        )
        self.conv_post_2 = nn.Sequential(
            nn.Conv2d(1, self.num_filters,
                      (filter_sizes[1], embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((K - filter_sizes[1] + 1, 1), 1)
        )
        self.BCE = nn.BCELoss()
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, om, data, lens):
        om = om.to(self.device)
        data = data.to(self.device)

        post_emb = self.embedding(om)
        post_emb = torch.unsqueeze(post_emb, dim=1)
        data_emb = self.embedding(data)
        data_emb = torch.reshape(data_emb, (-1, K, embedding_size))
        data_emb = torch.unsqueeze(data_emb, dim=1)


        # Original microblog
        om_1 = self.conv_om_1(post_emb)
        om_2 = self.conv_om_2(post_emb)
        h_pool_u = torch.cat((om_1, om_2), 3)
        h_pool_flat_u = torch.reshape(h_pool_u, [-1, self.num_filters_total])
        h_pool_flat_u = torch.unsqueeze(h_pool_flat_u, dim=0)

        # Posts
        post_1 = self.conv_post_1(data_emb)
        post_2 = self.conv_post_2(data_emb)
        post_h_pool = torch.cat((post_1, post_2), 3)
        data = torch.reshape(
            post_h_pool, (-1, max_len, self.num_filters_total))
        
        output, state = self.rnn(data, lens)
        state = torch.squeeze(state)
        h_pool_flat_u = torch.squeeze(h_pool_flat_u)
        h_pool_flat_u = torch.unsqueeze(h_pool_flat_u, 1)
        h_om = torch.cat(
            (output, h_pool_flat_u.repeat(1, max_len, 1)), -1)
        res = self.linear(h_om)
        res_softmax = F.softmax(res, dim=2)

        prediction = []
        for i in range(batch_size):
            temp_pred = torch.argmax(res_softmax[i, -1])
            for j in range(max_len):
                if torch.max(res_softmax[i, j]) >= alpha:
                    temp_pred = torch.argmax(res_softmax[i, j])
                    break
            prediction.append(temp_pred)
        prediction = torch.Tensor(prediction)
        res_softmax = torch.squeeze(res_softmax).to(torch.float64)
        return output, state, res_softmax, prediction

    def loss(self, labels, res_softmax):
        res_softmax = res_softmax.to('cpu')
        # Make it differentiable
        betas = []
        betas_idx = []
        for i in range(batch_size):
            beta_idx = max_len - 1
            for j in range(max_len):
                if torch.max(res_softmax[i][j]) >= alpha:
                    beta_idx = j
                    break
            betas.append((beta_idx + 1)/(max_len + 1))
            betas_idx.append(beta_idx)
        betas = torch.Tensor(betas)
        betas_idx = torch.Tensor(betas_idx).to(torch.int32)

        
        total_loss = torch.zeros(1)
        labels_extended = torch.unsqueeze(labels, dim=1)
        labels_extended = labels_extended.repeat(1, max_len).to(torch.float32)

        # O_time
        betas_sum = torch.sum(torch.log(betas))
        total_loss += (betas_sum * self.lambda1)

        # O_pred
        temp_pred_loss = torch.zeros(1)

        for i in range(batch_size):
            if betas[i] != 1:
                temp_pred_loss += F.cross_entropy(res_softmax[i, betas_idx[i]:, :], labels_extended[i, betas_idx[i]:len(res_softmax[i])].to(torch.long))

        total_loss += temp_pred_loss


        # O_diff
        temp_diff_loss = torch.zeros(1)
        for i in range(batch_size):
            if betas[i] != 1:
                o_i = F.cross_entropy(res_softmax[i, betas_idx[i]:, :], labels_extended[i, betas_idx[i]:len(res_softmax[i])].to(torch.long))
                y = labels[i]
                alpha_t = torch.Tensor([alpha])
                x1 = F.relu(torch.log(alpha_t) - o_i)
                x2 = F.relu(o_i - torch.log(torch.ones(1) - alpha_t))
                batch_loss = y * x1 + (1 - y) * x2
                
                temp_diff_loss += batch_loss

        total_loss += (temp_diff_loss * self.lambda0)
        return total_loss

    def step(self, batch, mode='train'):
        seq_data, labels = batch
        sequences = []
        lens = []
        oms = []

        for i in range(len(seq_data)):            
            om = seq_data[i]['seq'][0][:length]
            oms.append(om)
            temp_seq = []
            for j in seq_data[i]['seq']:
                temp_seq.append(j[:length])
            sequences.append(temp_seq)
            l = seq_data[i]['len']
            lens.append(l if l <= 100 else 100)




        lens = np.array(lens).astype(int)
        sequences = np.array(sequences).astype(int)
        labels = np.array(labels).astype(int)
        oms = np.array(oms).astype(int)


        # sort by lengths
        reverse_idx = np.argsort(-lens)

        sorted_length = lens[reverse_idx]
        sorted_sequnces = sequences[reverse_idx]
        sorted_labels = labels[reverse_idx]
        sorted_oms = oms[reverse_idx]
        sorted_length[0] = max_len

        oms  = torch.from_numpy(sorted_oms).to(torch.long)
        labels = torch.from_numpy(sorted_labels).to(torch.int32)
        sequences = torch.Tensor(sorted_sequnces.tolist()).to(torch.long)
        lens = torch.from_numpy(sorted_length).to(torch.int32)



        output, state, res_softmax, prediction = self.forward(oms, sequences, lens)
        loss = self.loss(labels, res_softmax)
        tn, fn, fp, tp = confusion_matrix(prediction.to(torch.int32), labels.to(
            torch.int32), num_classes=2, threshold=0.5).flatten()
        acc = (tp.item() + tn.item()) /             (tn.item() + fn.item() + tp.item() + fp.item())
        self.log(f'{mode}_loss', torch.round(loss.to(torch.float32)), on_step=True)
        self.log(f'{mode}_tn', float(tn.item()), on_step=True)
        self.log(f'{mode}_fn', float(fn.item()), on_step=True)
        self.log(f'{mode}_fp', float(fp.item()), on_step=True)
        self.log(f'{mode}_tp', float(tp.item()), on_step=True)
        self.log(f'{mode}_acc', float(acc), on_step=True)

        if mode == 'val':
            tune.report(loss=torch.round(loss.to(torch.float32)), acc=acc)

        return {
            'loss': loss,
            'acc': acc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn}

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(), 1e-3, eps=1e-10, weight_decay=0.9)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode='test')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode='val')



class SeqDataset(Dataset):
    def __init__(self, dataset , transform=None):
        self.dataset = dataset
        self.transform = transform
        super(SeqDataset, self).__init__()
    
    def __getitem__(self, item):
        sequence, label = self.dataset[item]
        if self.transform:
            sequence = self.transform(sequence)
        return sequence, label

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_len(self):
        return len(self.dataset[0][0])

    @property
    def input_size(self):
        return len(self.dataset[0][0][0])


def collate_fn(data):
    sequences, labels = zip(*data)

    return sequences, labels



class SeqDataset(Dataset):
    def __init__(self, dataset , transform=None):
        self.dataset = dataset
        self.transform = transform
        super(SeqDataset, self).__init__()
    
    def __getitem__(self, item):
        sequence, label = self.dataset[item]
        if self.transform:
            sequence = self.transform(sequence)
        return sequence, label

    def __len__(self):
        return len(self.dataset)

    @property
    def seq_len(self):
        return len(self.dataset[0][0])

    @property
    def input_size(self):
        return len(self.dataset[0][0][0])


def collate_fn(data):
    sequences, labels = zip(*data)

    return sequences, labels



max_epoch_num = 1
batch_size = 32


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="config file path")
parser.add_argument('--cuda', type=int, default=None, help="GPU number (default: None=CPU)")
parser.add_argument('--logdir', type=str, help="log directory")
parser.add_argument('--test', type=str, default=None, help="checkpoint path for test")

parser.add_argument('--input', type=str, help="input file path", required=True)
parser.add_argument('--learning-rate', type=float, default=0.2, metavar="0.2", help="learning rate for model")
parser.add_argument('--batch-size', type=int, default=64, metavar='32', help="batch size for learning")
parser.add_argument('--epoch', type=int, default=10, metavar="10", help="the number of epochs")


args = parser.parse_args()
dataset = args.input

def experiment(config):
    global alpha
    alpha = config['alpha']
    model = CEDCNN(config['lambda0'], config['lambda1'])
    train_data = [json.loads(d) for d in open(base_path + 'ced-inputs/' + dataset + f'/train_ced_{dataset}.txt', "rt").readlines()]
    val_data = [json.loads(d) for d in open(base_path + 'ced-inputs/' + dataset + f'/validation_ced_{dataset}.txt', "rt").readlines()]
    test_data = [json.loads(d) for d in open(base_path + 'ced-inputs/' + dataset+ f'/test_ced_{dataset}.txt', "rt").readlines()]
    train_dataset = SeqDataset(train_data)
    val_dataset = SeqDataset(val_data)
    test_dataset = SeqDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, collate_fn=collate_fn)

    name = f"dataset={dataset}-alpha={round(config['alpha'], 5)}-l0={round(config['lambda0'], 5)}-l1={round(config['lambda1'], 5)}"
    cp_name = checkpoint_path + f"/{name}"
    logger = loggers.CSVLogger(cp_name)
    checkpoint_callback = ModelCheckpoint(dirpath=cp_name, filename='{epoch}-{val_loss:.2f}', save_top_k=-1, monitor="val_loss")
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch_num, logger=logger, callbacks=[checkpoint_callback], log_every_n_steps=1, enable_progress_bar=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)




checkpoint_path = checkpoint_base_path + f"/{dataset}"
analysis = tune.run(
    experiment,
    name=dataset,
    num_samples=4,
    resources_per_trial={"cpu": 2, "gpu": 1/3},
    verbose=3,
    metric='loss',
    mode='min',
    scheduler=ASHAScheduler(),
    max_concurrent_trials=2,
    time_budget_s=timedelta(hours=9),
    config={
        "lambda0": tune.loguniform(1e-5, 1e-1),
        "lambda1": tune.loguniform(1e-5, 1e-1),
        "alpha": tune.grid_search([0.95, 0.975])
        }
    )

best_trial = analysis.get_best_trial("loss", "min", "last")
best_model = best_trial.config
print(f"Best hyperparameters for {dataset}: {best_model}")


