import torch
import os
import numpy as np
from torch.utils.data import Dataset
from itertools import chain
from torch.utils.data import DataLoader
import re
import random


def get_split_sentence(data,label,id2word,id2tag):
    entity=''
    res=[]
    for i in range(len(data)):
        tag = id2tag[label[i]]
        word = id2word[data[i]]
        if tag == 'B':
            entity = word
        elif tag == 'M' and len(entity) != 0 :
            entity+=word
        elif tag == 'E' and len(entity)!=0:
            entity+=word
            res.append(entity)
            entity=''
        elif tag == 'S':
            entity+=word
            res.append(entity)
            entity=''
        else :
            entity = ''
    return res
    

def get_right_count(true_label,pred_label,id2tag):
    true_label = [id2tag[i] for i in true_label]
    pred_label = [id2tag[i] for i in pred_label]
    assert len(true_label) == len(pred_label)
    true_len = 0
    pred_len = 0
    right_count = 0
    for i in range(len(true_label)):
        tl = true_label[i]
        pl = pred_label[i]
        if tl == 'S':
            true_len = 1
        else : 
            true_len += 1
        
        if tl == pl:
            pred_len += 1
        
        if tl == 'S' or tl == 'E':
            if pred_len == true_len :
                right_count += 1
            true_len = 0
            pred_len = 0
    return right_count
            
def print_result(id2word,id2tag,data,true_label,pred_label):
    true_result  = get_split_sentence(data,true_label,id2word,id2tag)
    pred_result = get_split_sentence(data,pred_label,id2word,id2tag)
    
    print("true result : ",' '.join(true_result))
    print("pred result : ",' '.join(pred_result))
    print("pred label : ",' '.join([id2tag[i] for i in pred_label]))


# split chinese string by "," or "。"
def split2short(pra):
    if isinstance(pra,str):
        result = pra.split('。')
        result[0:-1] = [t+'。' for t in result[0:-1]]
        
        sig=[]
        for i in result:
            i = i.split('，')
            i[0:-1] = [t+'，' for t in i[0:-1]]
            sig += i
        
        return sig
    elif isinstance(pra,list):
        result = [split2short(t) for t in pra]
        return list(chain(*result))
    else:
        raise TypeError

# remove all no chinese and replace by ' '
def RemoveNoChinese(x):
    if isinstance(x,str):
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese =  re.sub(pattern,' ',x)
        return chinese
    elif isinstance(x,list):
        return [RemoveNoChinese(t) for t in x]
    else :
        raise TypeError
        
def preprocess(text):
    # return text
    result = split2short(text)
    return result
    return RemoveNoChinese(result)
    
    
VVO = "<unknown>"
def GetVocab(data_path,load_file = True):
    if os.path.exists(data_path+"_vocab.npy") and load_file:
        vocab = np.load(data_path+"_vocab.npy",allow_pickle=True).item()
        return vocab
    else :
        with open(data_path,encoding = "utf8") as fp:
            source = fp.readlines()
            #source = RemoveNoChinese(source)
            oneline = str().join(source)
            oneline = str().join(oneline.split())
            oneline = list(set(oneline))
            oneline.sort()
            Vocab = {s:no for no,s in enumerate(oneline) } 
            Vocab[VVO] = len(Vocab)
            np.save(data_path+"_vocab.npy",Vocab)
            return Vocab

def collate_fn(data):
    # data (batch_size,(data,label,lens))
    data.sort(key=lambda x:x[2],reverse = True)
    sentence,label,length = [list(i) for i in zip(*data)]
    max_lens = length[0]
    for i,lens in enumerate(length):
        sentence[i] = np.append(sentence[i],np.zeros(max_lens-lens,dtype = np.int32))
        label[i] = np.append(label[i],np.zeros(max_lens-lens,dtype = np.int32))
    return torch.tensor(sentence).long(),torch.tensor(label).long(),torch.tensor(length).long()
 
class CRFDataSet(Dataset):
    def __init__(self,dataset_path,Vocab,Tag,load_file = True,random_block=False):
        self.Vocab = Vocab
        self.Tag = Tag
        self.data = None
        self.label = None
        self.length = None
        self.random_block = random_block
        if os.path.exists(dataset_path+".npy") and load_file:
            self.data,self.label,self.length = np.load(dataset_path+".npy",allow_pickle=True)
        else:
            with open(dataset_path,encoding="utf8") as fp:    
                self.data,self.label,self.length = self.process(fp.readlines())
                np.save(dataset_path+'.npy',(self.data,self.label,self.length))
        
    
    def randomBlock(self,data):
        se = random.randint(0,100)
        if(se>10):
            return data
        data = data[:]
        lens = 1 #random.randint(1,3)
        #if lens >= len(data):
        #    lens = 1
        chose = random.randint(0,len(data)-lens)
        for i in range(chose,chose+lens):
            data[i] = self.Vocab[VVO]
        return data
    
    def labelize(self,string):
        if len(string) == 1:
            return ['S']
        elif len(string) == 2:
            return ['B','E']
        else :
            label = ['M' for i in range(len(string)-2) ]
            return ['B']+label+['E']
    
    def process(self,source):
        source = preprocess(source)
        sequence_data = []
        sequence_label = []
        sequence_length = []
        for string in source:
            line = str().join(string.split())
            if len(line)>100:
                continue
            #if len(line)>100:
                #print(string)
            if line == "":
                continue
            sequence = np.array([self.Vocab[s] if s in self.Vocab else self.Vocab[VVO] for s in line])
            #for s in line:
            #    if s not in self.Vocab:
            #        print("unknown:",s)
            sequence_data.append(sequence)
            
            label = [ self.labelize(t) for t in string.strip().split()]
            label = [i for j in label for i in j]
            
            label = np.array([self.Tag[i] for i in label])
            assert len(label) == len(sequence)
            sequence_label.append(label)
            sequence_length.append(len(label))
            
        return sequence_data,sequence_label,sequence_length
    def __len__(self):
        assert len(self.label) == len(self.data)
        return len(self.data)
    def __getitem__(self, index):
        retdata = self.data[index]
        if self.random_block:
            retdata = self.randomBlock(retdata)
        return retdata,self.label[index],self.length[index]

if __name__ == "__main__":
    fp = open("./pku_data/pku_data/minitest.txt",encoding='utf-8')
    lines = fp.readlines()
    lines = preprocess(lines)
    for string in lines:
        line = str().join(string.split())
        print(line)
    