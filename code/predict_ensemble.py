import torch
import torch.nn.functional as F
from torchtext.legacy import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
import torch.utils.data as Data
from star_transformer import STSeqCls_Pred


def make_data(data, vocab_dict):
    pep_inputs, hla_inputs, labels = [], [], []
    pep_lens = []
    for pep in data.text:
        pep_lens.append(len(pep))
        tokenizer = lambda sent: [x for x in sent if x != " "]
        tokenized = tokenizer(pep)
        tokenized += ['<pad>'] * (50 - len(tokenized))
        pep_input = [[vocab_dict[t] for t in tokenized]]

        pep_inputs.extend(pep_input)

    return torch.LongTensor(pep_inputs), torch.LongTensor(pep_lens)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, pep_lens):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.pep_lens = pep_lens

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.pep_lens[idx]


def predict_dataloader(test_df,vocab):

    pep_inputs, pep_lens = make_data(test_df, vocab)
    data_loader = Data.DataLoader(MyDataSet(pep_inputs, pep_lens), 512, shuffle = False, num_workers = 0)
    return test_df, pep_inputs, pep_lens, data_loader


def predict_mul_batch(model, test_iterator, device):
    '''
        df DataFrame: peptide  allele
                AAAAAAA  HLA-A0201
    '''
    proba_list1 = []
    proba_list2 = []
    proba_list3 = []
    proba_list4 = []

    df_output_ensemble = pd.DataFrame()
    model.eval()
    with torch.no_grad():
        for x,l in test_iterator:
            if torch.cuda.is_available():
                x = x.to(device)
                l = l.to(device)
            y_pred = model(x,l)
            proba = F.softmax(y_pred, dim=1)
            proba_list1.extend(proba.data[:, 0].cpu().numpy())
            proba_list2.extend(proba.data[:, 1].cpu().numpy())
            proba_list3.extend(proba.data[:, 2].cpu().numpy())
            proba_list4.extend(proba.data[:, 3].cpu().numpy())
    df_output_ensemble['proba1'] = proba_list1
    df_output_ensemble['proba2'] = proba_list2
    df_output_ensemble['proba3'] = proba_list3
    df_output_ensemble['proba4'] = proba_list4
    df_output_ensemble['proba'] = df_output_ensemble.apply(lambda x: x['proba1']+x['proba2']+x['proba3']+x['proba4'], axis=1)
    return df_output_ensemble['proba']


def st_hla_predict(df, device):
    df2 = df[['label','length']]
    peptides = df['peptide']
    hla = df['allele']
    df_candicated_all_hla = pd.DataFrame()

    df = pd.DataFrame({'peptide':peptides,'allele':hla})

    df['allele'] = df['allele'].map(lambda x: x.replace('*','').replace(':',''))
    allele = pd.read_csv('../data/other/HLAI_pseudosequences_34mer.csv')

    df = df.merge(allele, how='left')
    df['text'] = df.apply(lambda x: str(x['peptide'])+','+str(x['pseudosequence']),axis=1)

    vocab_dict_list = ['../model/vocab1.npy',
    '../model/vocab2.npy',
    '../model/vocab3.npy',
    '../model/vocab4.npy',
    '../model/vocab5.npy',]
    model_list = ['../model/model1.pth',
    '../model/model2.pth',
    '../model/model3.pth',
    '../model/model4.pth',
    '../model/model5.pth']
    df_ret = pd.DataFrame()

    for i,m in enumerate(model_list):
        vocab_dict = np.load(vocab_dict_list[i], allow_pickle=True).item()
        _,_,_,data_loader = predict_dataloader(df[['text']], vocab_dict) #vocab=vocab_dict
        print('model ',i, m)
        model = STSeqCls_Pred((23, 100), 6, hidden_size=100, num_layers=3, num_head=5, max_len=50,cls_hidden_size=600,dropout=0.1).to(device)
        model.load_state_dict(torch.load(m))
        presentation_score = predict_mul_batch(model, data_loader, device)
        df_ret[f'proba{i}'] = presentation_score

    print('ensemble output: \n',df_ret)
    df_ret['mean_proba'] = df_ret.apply(lambda x:x.mean(),axis=1)
    df_ret = df_ret.round(4)
    assert len(df) == len(df_ret)
    df['presentation'] = df_ret['mean_proba']


    transfer_vocab_dict_list = ['../model/vocab2.npy']
    transfer_model_list = ['../model/transfer_model2.pth']
    transfer_df_ret = pd.DataFrame()
    for i,m in enumerate(transfer_model_list):
        transfer_model = STSeqCls_Pred((23, 100), 6, hidden_size=100, num_layers=3, num_head=5, max_len=50,cls_hidden_size=600,dropout=0.1).to(device)
        transfer_model.load_state_dict(torch.load(m))

        vocab_dict = np.load(transfer_vocab_dict_list[i], allow_pickle=True).item()
        _, _, _, transfer_data_loader = predict_dataloader(df[['text']], vocab_dict) #vocab=vocab_dict

        neoantigen_score = predict_mul_batch(transfer_model, transfer_data_loader, device)
        transfer_df_ret[f'proba{i}'] = neoantigen_score
    transfer_df_ret['mean_proba'] = transfer_df_ret.apply(lambda x:x.mean(),axis=1)
    transfer_df_ret = transfer_df_ret.round(4)
    print('tranfer output\n: ',transfer_df_ret)
    assert len(df) == len(transfer_df_ret)    
    df['neoantigen'] = transfer_df_ret['mean_proba']
    
    df_candicated_all_hla = df_candicated_all_hla.append(pd.concat([df,df2],axis = 1))

    del df_candicated_all_hla['text']
    df_candicated_all_hla.to_csv('./result.csv',index=False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('../data/test_set/external_set/HLA-A0101.csv')
    st_hla_predict(df, device)


