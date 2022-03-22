import torch
from torch import nn
import torch.optim as optim
import numpy as np 

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

params={
        "embedding_size":100,
        "encode_size":200,
        'hidden_size':50,
        }
START_TAG = "<start>"
END_TAG = "<end>"
PAD_TAG = "<pad>"
class BiLSTM_CRF(nn.Module):
    def __init__(self,Vocab,Tag,device):
        super(BiLSTM_CRF,self).__init__()
        
        self.vocab_size = len(Vocab)
        self.tag_num = len(Tag)
        self.Tag = Tag
        self.device = device
        self.word_embeddings = nn.Embedding(self.vocab_size,params["embedding_size"])
        self.encoder = nn.LSTM(params["embedding_size"],params["encode_size"]//2,
                               num_layers=1,bidirectional=True,
                               batch_first = True)
        # word => embedding => score for every positon => CRF
        self.encode2hidden = nn.Linear(params["encode_size"],params["hidden_size"],bias = True)
        self.hidden2tag = nn.Linear(params['hidden_size'],self.tag_num,bias = True)
        self.DropOutLayer = nn.Dropout(0.5)
        self.ReLU = nn.ReLU()
        # some for CRF, transitions[i,j] means trans from i to j
        self.transitions = nn.Parameter(torch.randn(self.tag_num,self.tag_num))
        
        # force disable the trans from any to start or end to any
        # consider transiton is added to score,which expected to be zero
        self.transitions.data[:,Tag[START_TAG]] = -1e6
        self.transitions.data[Tag[END_TAG],:] = -1e6
        
    def _get_probs(self,X,length):
        b_size = len(length)
        
        h_0 = torch.randn(2, b_size,params["encode_size"] // 2).to(self.device)
        c_0 = torch.randn(2, b_size,params["encode_size"] // 2).to(self.device)
        
        X_embeded = self.word_embeddings(X)
        # X_embeded : (b,max_len,embed_size)
        X_packed = pack_padded_sequence(X_embeded,length,batch_first = True)
        encode, _ = self.encoder(X_packed,(h_0,c_0))
        X_unpacked, _ = pad_packed_sequence(encode, batch_first=True)
        # X_unpacked : (batch,len,embeding_size)
        #return self.hidden2tag(self.encode2hidden(X_unpacked))
        x = self.ReLU(self.encode2hidden(X_unpacked))
        x = self.DropOutLayer(x)
        logits = self.hidden2tag(x)
        return logits
        
        
    def neg_log_likelihood(self,X,y,length):
        # X is train data, y is label
        # let. min. -log(p(y|X)) = -score(X,y) + log(sum(exp(s(X,y)))) 
        # X: (batch,sequence_lens,vocab_size) one-hot 
        # y: (batch,sequence_lens,)
        # length :(batch)
        # word_embeddings(X) : (sequence_lens,vocab_size)=>
        #                           (sequence_lens,embedding_size)
        # y_pred : (sequence_lens,tag_num)
        probs = self._get_probs(X,length)
        # score(X,y)
        expected_score = self._score_for_sentence(probs,y,length)
        total_score = self._score_total(probs,length)
        
        return torch.mean(total_score - expected_score)
    def _score_for_sentence(self,probs,sentence_tag,length):
        # culculate score when chosing tags as sentence_tag
        # probs : (batch,max_size,tag_num)
        # sentence_tag : (batch,senquence_size) content is the tag idx
        
        # score(X,y) = sum(A_{y_i,y_i+1}) + sum(P_{i,y_i}) 
        # first is tansitions from tag_yi to tag_yi+1
        # second is probs when tag is y_i
    
        #sentance_tag is tensor that first is start ,last is end
        b_size = len(length)
        self.start_tag = torch.tensor([self.Tag[START_TAG]]*b_size,
                                      dtype = torch.long).to(self.device)
        self.start_tag = self.start_tag.reshape((-1,1))
        
        sentence_tag = torch.cat((self.start_tag[:b_size],sentence_tag),1)
        score = torch.zeros(b_size).to(self.device)
        
        max_length = length[0]
        for i in range(max_length):
            ran = torch.ge(length,i+1).sum()
            # [0,ran)  batch 是当前有效的
            score[:ran] += self.transitions[sentence_tag[:ran,i],sentence_tag[:ran,i+1]] \
                        + probs[list(range(ran)),i,sentence_tag[:ran,i + 1]]
        #for i,prob in enumerate(probs):
        #    score += self.transitions[sentence_tag[i],sentence_tag[i + 1]] + prob[sentence_tag[i + 1]]
        
        return score + self.transitions[sentence_tag[list(range(b_size)),length],self.Tag[END_TAG]]
    def _score_total(self,probs,length):
        b_size = len(length)
        alpha = torch.full((b_size,self.tag_num),-1e6).to(self.device)
        alpha[:,self.Tag[START_TAG]] = 0.0
        
        # alpha (batch,tagnum),tansition(tagnum,tagnum)
        # prob(batch,tagnum)
        max_length = length[0]
        for i in range(max_length):
            ran = torch.ge(length,i+1).sum()
            
            prob = probs[:ran,i,:]

            temp = alpha[:ran].reshape(ran,self.tag_num,1) \
                        + self.transitions.reshape(1,self.tag_num,self.tag_num)\
                        + prob.reshape(ran,1,self.tag_num)
            alpha[:ran] = self.log_sum_exp(temp).reshape(ran,-1)
            
        #for prob in probs:
        # boardcast alpha(tag_num,1) + A(tag_num,tag_num) + P(tag_num,)
        #    alpha = alpha.reshape(-1,1) + self.transitions + prob.reshape(-1)
            # every column is a ahpha_i
        #    alpha =self.log_sum_exp(alpha)    # means get anser by columns
        # add end state
        # alpha : (batch,tagnum)
        # transition:(tagnum)
        alpha = alpha + self.transitions[:,self.Tag[END_TAG]].reshape(1,-1)
        alpha = self.log_sum_exp(alpha).reshape(-1)
        return alpha
    
    
    def log_sum_exp(self,x):
        # x ran,tag_num,tag_num
        x_max = x.max(dim=1,keepdim=True).values   #for avoid inf
        return (x-x_max).exp().sum(axis=1,keepdim=True).log()+x_max
    def forward(self,X,lengths):
        probs = self._get_probs(X,lengths)
        scores=[]
        paths = []
        #for prob in probs:
        #    score,path = self.Viterbi(prob)
        #    scores.append(score)
        #    paths.append(path)
        scores,paths = self.Viterbi(probs,lengths)
        return scores,paths
    
    def Viterbi(self,probs,lengths):
        b_size = len(lengths)
        alpha = torch.full((b_size,self.tag_num),-1e6).to(self.device)
        alpha[:,self.Tag[START_TAG]] = 0.0
        best_idxs=[[] for i in range(b_size)]
        
        max_length = lengths[0]
        for i in range(max_length):
            ran = torch.ge(lengths,i+1).sum()
            prob = probs[:ran,i,:]
            temp = alpha[:ran].reshape(ran,self.tag_num,1)\
                    +self.transitions.reshape(1,self.tag_num,self.tag_num)\
                    +prob.reshape(ran,1,self.tag_num)

            idx = torch.argmax(temp,dim=1)
            for i in range(ran):
                best_idxs[i].append(idx[i])

            temp,_ = temp.max(axis=1,keepdim=False)
            alpha[:ran] = temp
        alpha = alpha.reshape(b_size,self.tag_num) \
                    + self.transitions[:,self.Tag[END_TAG]].reshape(1,-1)
        best_score = alpha.max(axis = 1).values.reshape(-1)
        
        best_paths = []
        for i in range(b_size):
            best_path=[]
            best_path.append(alpha[i].argmax().reshape(-1))
            for idx in reversed(best_idxs[i]):
                pre_best = best_path[-1]
                best_path.append(idx[pre_best])
            best_path.pop()  # remove start idx
            best_path.reverse()
            best_paths.append(torch.tensor(best_path))
        return best_score,best_paths
