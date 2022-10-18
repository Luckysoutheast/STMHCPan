# utils.py

import torch
from torchtext.legacy import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

class Dataset(object):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = None

    def get_pandas_df(self, filename):
        df = pd.read_csv(filename)
        data_text = df.apply(lambda x: x['peptide']+','+x['HLA_sequence'],axis=1)
        data_label = df['label']
        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    def load_data(self, train_file, val_file):

        tokenizer = lambda sent: [x for x in sent if x != " "]
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=False, batch_first=True, include_lengths=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)#self.get_pandas_df(train_file)
        # train_df.to_csv('./train_test.csv',index=False)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        val_df = self.get_pandas_df(val_file)#self.get_pandas_df(test_file)
        val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        val_data = data.Dataset(val_examples, datafields)
        
        
        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab.stoi
        np.save('./vocab.npy', TEXT.vocab.stoi)

        self.train_iterator, self.val_iterator = data.BucketIterator.splits(
            (train_data, val_data),
            batch_sizes=(self.batch_size,self.batch_size),
            sort_key=lambda x: len(x.text), # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=False,
            repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
        )
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} validation examples".format(len(val_data)))
        print ("vocab {}".format(TEXT.vocab.itos))

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

def Adam_LLRD(model):

    opt_parameters = []    
    named_parameters = list(model.named_parameters()) 

    no_decay = ["bias", "norm"]
    lr = 1e-4

    params_0 = [p for n,p in named_parameters if "cls.fc" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "cls.fc" in n
                and not any(nd in n for nd in no_decay)]

    fc_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(fc_params)

    fc_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(fc_params)   
    
    lr *= 0.9

    for layer in range(2,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.star_att.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.star_att.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       

   
    lr *= 0.9    
    for layer in range(2,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.ring_att.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.ring_att.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)       


    lr *= 0.9
    params_0 = [p for n,p in named_parameters if "emb_fc" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "emb_fc" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)        
    
    lr *= 0.9
    params_0 = [p for n,p in named_parameters if "embedding" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embedding" in n
                and not any(nd in n for nd in no_decay)]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01} 
    opt_parameters.append(embed_params)    

    return opt_parameters


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass




def get_score(y_ture, y_pred, y_score):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    f1 = f1_score(y_ture, y_pred) * 100
    p = precision_score(y_ture, y_pred) * 100
    r = recall_score(y_ture, y_pred) * 100
    auc = roc_auc_score(y_ture, y_score)*100
    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2), reformat(auc, 2)

def get_score_mul(y_ture, y_pred, y_score):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    f1 = f1_score(y_ture, y_pred, average='weighted') * 100
    p = precision_score(y_ture, y_pred, average='weighted') * 100
    r = recall_score(y_ture, y_pred, average='weighted') * 100
    try:
        auc = roc_auc_score(label_binarize(y_ture,np.arange(6)), y_score, multi_class='ovo')*100
    except:
        auc = 0
    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2), reformat(auc, 2)


def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))