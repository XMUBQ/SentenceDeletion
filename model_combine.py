from utils import *

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()

        self.context_encoder = nn.LSTM(config['embedding_dim'], config['hidden_dim'], config['num_layers'],
                              batch_first=True, bidirectional=config['bidirectional']) #BiLSTM(config)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.method=config['method']
        self.discourse_size=9
        self.tagset_size=2

        self.inner_pred_nn = nn.Linear((config['hidden_dim']*2), config['embedding_dim'])
        if self.method == 'joint_learning':
            self.pred_nn = nn.Linear((config['hidden_dim'] * 6), self.tagset_size)
            self.disc_pred_nn=nn.Linear((config['hidden_dim']*6),self.discourse_size)
        elif self.method == 'concat':
            self.pred_nn = nn.Linear((config['hidden_dim'] * 6)+self.discourse_size, self.tagset_size)
        elif self.method == 'base':
            self.pred_nn = nn.Linear((config['hidden_dim'] * 6), self.tagset_size)

        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear((config['hidden_dim']*2), (config['hidden_dim']*2))
        self.ws2 = nn.Linear((config['hidden_dim']*2), 1)
        self.softmax = nn.Softmax(dim=1)
        self.discourse_encoder = nn.LSTM(config['embedding_dim'], config['hidden_dim'], config['num_layers'],
                              batch_first=True, bidirectional=config['bidirectional'])
        
        self.number_of_cats=9
        self.maxpoolRange=2
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size,self.tagset_size))

        self.init_weights()

    def init_weights(self):
        for name, param in self.discourse_encoder.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)

        for name, param in self.context_encoder.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)

        nn.init.xavier_uniform_(self.pred_nn.state_dict()['weight'])
        self.pred_nn.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ws2.state_dict()['weight'])
        self.ws2.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.inner_pred_nn.state_dict()['weight'])
        self.inner_pred_nn.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.pred_nn.state_dict()['weight'])
        self.pred_nn.bias.data.fill_(0)

        if self.method == 'joint_learning':
            nn.init.xavier_uniform_(self.disc_pred_nn.state_dict()['weight'])
            self.disc_pred_nn.bias.data.fill_(0)

      
    def forward(self, x, discourse_distribution):
        torch.cuda.empty_cache()
        input_feat = self.tokenizer.batch_encode_plus(x, max_length=512,
                                                 padding='longest',  # implements dynamic padding
                                                 truncation=True,
                                                 return_tensors='pt',
                                                 return_attention_mask=True,
                                                 return_token_type_ids=True
                                                 )
        if cuda_available:
            input_feat['attention_mask'] = input_feat['attention_mask'].cuda()
            input_feat['input_ids'] = input_feat['input_ids'].cuda()
        with torch.no_grad():
            outputs = self.bert(input_feat['input_ids'], attention_mask=input_feat['attention_mask'])

        disc_feat=discourse_distribution
        sentence=outputs.pooler_output

        ctxt_mask_2=torch.tensor([[1]]*sentence.shape[0])
        if cuda_available:
            ctxt_mask_2=ctxt_mask_2.cuda()

        outp_ctxt=sentence.unsqueeze(1)
        outp_2,_ = self.context_encoder(outp_ctxt)
        
        self_attention = torch.tanh(self.ws1(self.drop(outp_2)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()
        self_attention = self_attention + -10000*(ctxt_mask_2 == 0).float()
        self_attention = self.softmax(self_attention)

        sent_encoding = torch.sum(outp_2*self_attention.unsqueeze(-1), dim=1)
        _inner_pred = torch.tanh(self.inner_pred_nn(self.drop(sent_encoding)))
        inner_pred, (hidden, ctxt) = self.discourse_encoder.forward(_inner_pred[None, :, :])
        inner_pred = inner_pred.squeeze()
        
        AllBefore=torch.zeros(inner_pred.shape[0],inner_pred.shape[1])
        if cuda_available:
            AllBefore=AllBefore.cuda()
        AllAfter=torch.zeros(inner_pred.shape[0],inner_pred.shape[1])
        if cuda_available:
            AllAfter=AllAfter.cuda()
        for i in range(inner_pred.shape[0]):
            before=torch.zeros(1,inner_pred.shape[1])
            if cuda_available:
                before=before.cuda()
            if not(i==0):
                start=i-self.maxpoolRange if i>=self.maxpoolRange else 0
                beforeMaxpool=nn.MaxPool1d(i-start)
                before=beforeMaxpool(inner_pred.unsqueeze(0).permute(0,2,1)[:,:,start:i]).squeeze(2)
            AllBefore[i]=before

            after=torch.zeros(1,inner_pred.shape[1])
            if cuda_available:
                after=after.cuda()
            if not(inner_pred.shape[0]-i-1==0):
                end=1+i+self.maxpoolRange if 1+i+self.maxpoolRange<=inner_pred.shape[0] else inner_pred.shape[0]
                afterMaxpool=nn.MaxPool1d(end-i-1)
                after=afterMaxpool(inner_pred.unsqueeze(0).permute(0,2,1)[:,:,i+1:end]).squeeze(2)
            AllAfter[i]=after
        pre_pred=torch.cat((AllBefore,AllAfter,inner_pred),dim=1)#pre_pred = torch.tanh(self.pre_pred(self.drop(inner_pred)))

        if self.method == 'joint_learning':
            pred = self.pred_nn(self.drop(pre_pred))
            disc_pred=self.disc_pred_nn(self.drop(pre_pred))
        elif self.method == 'concat':
            pre_pred=torch.cat((self.drop(pre_pred),disc_feat),dim=1)
            pred = self.pred_nn(pre_pred)
            disc_pred = None
        elif self.method == 'base':
            pred = self.pred_nn(self.drop(pre_pred))
            disc_pred = None

        return pred, disc_pred