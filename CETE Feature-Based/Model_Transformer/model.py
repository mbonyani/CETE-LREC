# Model.py
import numpy as np
from torch.nn import init
from torch.functional import norm
import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings,PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from encoder_cross import EncoderCross, EncoderLayerCross
from feed_forward import PositionwiseFeedForward
#from Pytorch.Model_Transformer.imap_qa import calc_one_map
#from Pytorch.Model_Transformer.utils import Dataset
from utils import *
from train import return_file_name, return_emb1,return_emb2

from config import Config

config=Config()
#EmbeddingType="Contextualized" #for CETE
#Type="Baseline" #for GloVe


class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self,in_ch,out_ch,k):
        super().__init__()
        if(k==1):
            self.depth_conv=nn.Identity()
        else:
            self.depth_conv=nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k//2
                )
        self.pointwise_conv=nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )
    def forward(self,x):
        out=self.pointwise_conv(self.depth_conv(x))
        return out
    
class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):


        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1=Depth_Pointwise_Conv1d(h * d_v, d_model,1)
        self.conv3=Depth_Pointwise_Conv1d(h * d_v, d_model,3)
        self.conv5=Depth_Pointwise_Conv1d(h * d_v, d_model,5)
        self.dy_paras=nn.Parameter(torch.ones(3))
        self.softmax=nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        #Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2=v.permute(0,1,3,2).contiguous().view(b_s,-1,nk) #bs,dim,n
        self.dy_paras=nn.Parameter(self.softmax(self.dy_paras))
        out2=self.dy_paras[0]*self.conv1(v2)+self.dy_paras[1]*self.conv3(v2)+self.dy_paras[2]*self.conv5(v2)
        out2=out2.permute(0,2,1) #bs.n.dim

        out=out+out2
        return out

class Transformer(nn.Module):
    def __init__(self, config, src_vocab, target_vocab,s_v,t_v, u):
        super(Transformer, self).__init__()
        self.config = config

        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        
        #attn = MultiHeadedAttention(h, d_model)
        attn = MUSEAttention(d_model=d_model, d_k=d_model, d_v=d_model, h=h)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        #attncross = MultiHeadedAttention(h, d_model*2)
        attncross = MUSEAttention(d_model=d_model*2, d_k=d_model*2, d_v=d_model*2, h=h)

        ffcross = PositionwiseFeedForward(d_model*2, d_ff, dropout)
        positioncross = PositionalEncoding(d_model*2, dropout)
        
        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.encoder_cross = EncoderCross(EncoderLayerCross((config.d_model)*2, deepcopy(attncross), deepcopy(ffcross), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab, s_v, u), deepcopy(position)) #Embeddings followed by PE
        #self.src_embed.weight.data.copy_(src_vocab.vectors)
        self.target_embed = nn.Sequential(Embeddings(config.d_model, target_vocab,t_v, u), deepcopy(position))
        #self.target_embed.weight.data.copy_(target_vocab.vectors)
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        self.sigmoid=nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = nn.Softmax()

        # Softmax non-linearity

    def forward(self, x1, x2, idx,  type="default"):
        #print(x1)
        #print(idx+1)

        #print(label)
        idx=idx.data.cpu().numpy()
        if config.EmbeddingType=="Contextualized":
            embedded_sents1 = return_emb1(int(idx[0]),idx.size)
            embedded_sents2 = return_emb2(int(idx[0]),idx.size)
            embedded_sents1= torch.from_numpy(embedded_sents1).float().cuda()
            embedded_sents2 = torch.from_numpy(embedded_sents2).float().cuda()
        else:
            embedded_sents1 = self.src_embed(x1.permute(1, 0))  # shape = (batch_size, sen_len, d_model)
            embedded_sents2 = self.target_embed(x2.permute(1, 0))  # shape = (batch_size, sen_len, d_model)

        encoded_sents1 = self.encoder(embedded_sents1, embedded_sents1)
        encoded_sents2 = self.encoder(embedded_sents2, embedded_sents2)



        final_feature_map1 = torch.mean(encoded_sents1, 1)
        final_feature_map2 = torch.mean(encoded_sents2, 1)




        final_out1 = final_feature_map1
        final_out2 = final_feature_map2
        
        if final_out1.shape[0]!=final_out2.shape[0]:
           a = min(final_out1.shape[0] ,final_out2.shape[0])
           final_out1 = final_feature_map1[:a]
           final_out2 = final_feature_map2[:a]
           
    
        output=self.cos(final_out1 , final_out2)

        # for Ablation studies, uncomment the following (line 88 to 99)

        #final_feature_map1 = torch.mean(embedded_sents1, 1)
        #final_feature_map2 = torch.mean(embedded_sents2, 1)
        #final_out1 = final_feature_map1
        #final_out2 = final_feature_map2
        #comp = self.cos(final_out1, final_out2)

        #j=0
        #print(output.size(),"output")
        #print(comp.size(),"comp")
        #for i in comp:
        #    output[j] = comp[j]
        #    j = j + 1


        ''' '''

        return output
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, test_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        self.train()
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            #print(i)
            #print(batch)
            self.optimizer.zero_grad()

            if torch.cuda.is_available():
                x1 = batch.text1.cuda()
                x2 = batch.text2.cuda()
                #x=batch.text.cuda()
                y = (batch.label ).type(torch.cuda.FloatTensor)
                idx = (batch.index).type(torch.cuda.FloatTensor)
            else:
                x1 = batch.text1
                x2 = batch.text2
                y = (batch.label ).type(torch.FloatTensor)
                idx = (batch.index).type(torch.FloatTensor)
            #print(x2)

            y_pred = self.__call__(x1, x2, idx, y)
            #print(y)
            #
            #if(self.config.output_size>=2):
            #    y_pred=torch.max(y_pred, 1)[0]
            #y_pred = torch.max(y_pred, 1)[0]
            #y_pred=y_pred.to(torch.float32)
            #print(y_pred)
            #print(y)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                #print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                #print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
        trainfilename,testfilename,validfilename = return_file_name()
        print("Evaluating Epoch")
        '''
        val_map = evaluate_model(self, val_iterator, filename = validfilename)
        train_map = evaluate_model(self, train_iterator, filename = trainfilename)
        print("val map \t" + str(val_map))
        print("train map \t" + str(train_map))
        return train_losses, val_map
        '''
        config=Config()
        val_accuracy,v_c = evaluate_model(self, val_iterator, filename = validfilename)
        #train_accuracy = evaluate_model(self, train_iterator, filename = trainfilename)
        test_accuracy,t_c = evaluate_model(self, test_iterator, filename = testfilename)
        print("validation \t"+str(val_accuracy*config.multiplyby))
        #print("training \t" + str(train_accuracy*config.multiplyby))
        print("test \t" + str(test_accuracy*config.multiplyby))
        return train_losses, val_accuracy, v_c


