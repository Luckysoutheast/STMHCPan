import logging
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import classification_report
from utils import *
from star_transformer import STSeqCls

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s',filename='STSeqCls-IEDB.log')

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

# build trainer



clip = 5.0
epochs = 100
early_stops = 10
log_interval = 500
batch_size = 512

dataset = Dataset(batch_size)
train_file = '../data/train_set/IEDB/train_data.csv'
val_file = '../data/train_set/IEDB/val_data.csv'
dataset.load_data(train_file, val_file=val_file)


logging.info("vocab {}".format(dataset.vocab))

model = STSeqCls((23, 100), 6, hidden_size=100, num_layers=3, num_head=5, max_len=50,cls_hidden_size=600,dropout=0.1)

logging.info("model {}".format(model))
if use_cuda:
    model.to(device)
class Trainer():
    def __init__(self, model):
        self.model = model
        self.report = True

        self.train_data = dataset.train_iterator#get_examples(train_data, vocab)
        self.batch_num = len(self.train_data)
        self.dev_data = dataset.val_iterator#get_examples(dev_data, vocab)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = ['Positive','Positive-high','Positive-intermediate','Positive-low','Negative','Negative-random']

        # optimizer
        self.optimizer = ScheduledOptim(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 2, 100, 8000)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.best_dev_auc = 0
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training...')
        dev_f1s=[]
        dev_aucs = []
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)

            dev_f1, dev_auc = self._eval(epoch)
            dev_f1s.append(dev_f1)
            dev_aucs.append(dev_auc)

            if self.best_dev_auc <= dev_auc:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_auc, dev_auc))
                torch.save(self.model.state_dict(), f'./work/model_f1_{dev_auc}_{epoch}.pth')

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.best_dev_auc = dev_auc
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - early_stops, self.best_train_f1, self.best_dev_f1))


                    self.last_epoch = epoch
                    break
        #pd.DataFrame({'f1':dev_f1s, 'auc':dev_aucs}).to_csv('./dev_f1s.csv',index=False)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        y_score = []
        for batch_data in self.train_data:
            torch.cuda.empty_cache()
            x = batch_data.text[0]
            l = batch_data.text[1]
            batch_labels = batch_data.label
            if use_cuda:
                x = x.to(device)
                l = l.to(device)
                batch_labels = batch_labels.to(device)
            batch_outputs = self.model(x,l)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())
            y_score.extend(F.softmax(batch_outputs,dim=1).detach().cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            self.optimizer.step_and_update_lr()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer._optimizer.param_groups[0]['lr']
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / log_interval,
                        elapsed / log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1, auc = get_score_mul(y_true, y_pred, y_score)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f} | AUC {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses,
                                                                                  during_time, auc))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch):
        self.model.eval()
        start_time = time.time()
        y_pred = []
        y_score = []
        y_true = []
        y_score = []
        with torch.no_grad():
            for batch_data in self.dev_data:
                torch.cuda.empty_cache()
                x = x = batch_data.text[0]
                l = batch_data.text[1]
                batch_labels = batch_data.label
                if use_cuda:
                    x = x.to(device)
                    l = l.to(device)
                    batch_labels = batch_labels.to(device)
                batch_outputs = self.model(x,l)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())
                y_score.extend(F.softmax(batch_outputs,dim=1).detach().cpu().numpy().tolist())

            score, f1, auc = get_score_mul(y_true, y_pred, y_score)

            during_time = time.time() - start_time
            

            logging.info(
                '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f} | AUC {:.2f}'.format(epoch, score, f1,
                                                                          during_time, auc))
            if set(y_true) == set(y_pred) and self.report:
                report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                logging.info('\n' + report)

        return f1, auc

# train
trainer = Trainer(model)
trainer.train()
