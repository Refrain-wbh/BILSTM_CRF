
import sys
from tqdm.notebook import tqdm
import argparse
from data_process import *
from crf import * 
import time 

# torch.manual_seed(1)
parser = argparse.ArgumentParser(description='参数说明')
parser.add_argument('--train',action='store_true', help='训练模式')
parser.add_argument('--test',action = 'store_true', help='测试模式')
parser.add_argument('--train_path',type = str,
                    default = './data/train/merge_trainning.utf8',help='指定训练文件路径，注：字典来源于训练文件')
parser.add_argument('--test_path',type = str,
                    default = './data/test/merge_test_gold.utf8',help='指定测试文件路径')
parser.add_argument('--model',type = str,
                    default = None,help='指定模型加载路径（用于训练，以及保存，默认保存在./model/model.pth）')
parser.add_argument('--predict_model',type = str,
                    default = './model/p_model.pth',help='指定模型加载路径（用于测试）')

parser.add_argument('--epoch',type = int,
                    default = 10,help='指定训练轮数')
parser.add_argument('--batch_size',type = int,
                    default = 500,help='指定batch大小')

args = parser.parse_args()



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Tag = {
       START_TAG:0,
       END_TAG:1,
       "S":2,
       "B":3,
       "M":4,
       "E":5
       }
Id2Tag = {v:k for k,v in Tag.items()}

Vocab = GetVocab(args.train_path)
Id2Word = {v:k for k,v in Vocab.items()}

traindataset = CRFDataSet(args.train_path,Vocab,Tag,True,True)
train_loader = DataLoader(traindataset,batch_size=args.batch_size,shuffle=True,drop_last=False,collate_fn=collate_fn)

testdataset = CRFDataSet(args.test_path,Vocab,Tag)
test_loader = DataLoader(testdataset,batch_size=args.batch_size,shuffle=True,drop_last=False,collate_fn=collate_fn)

model = BiLSTM_CRF(Vocab, Tag,DEVICE)
optimizer = optim.Adam(model.parameters(),lr=0.005, weight_decay=1e-4)

model.to(DEVICE)



def train():
    start = time.time()
    allnum = len(train_loader)
    model.train()
    allloss = 0
    for i,(data,label,length) in enumerate(train_loader):
        model.zero_grad()
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        loss = model.neg_log_likelihood(data,label,length)
        allloss += loss
        if i % 10 == 0:
            pretime = time.time()
            print("(%d/%d) use time:%f,predict time:%f,loss:%f"%(i,allnum,pretime-start,
                            float(pretime-start)/(i+1)*allnum,
                            allloss/10))
            allloss = 0

        loss.backward()
        optimizer.step()
        
def test():
    with torch.no_grad():
        model.eval()
        TP=0
        FP=0
        FN=0
        for datas,true_labels,lengths in test_loader:
            datas =datas.to(DEVICE)
            scores,pred_labels = model(datas,lengths)
            
            datas=datas.cpu().numpy()
            true_labels = true_labels.numpy()
            pred_labels = [i.numpy() for i in pred_labels] 
            
            for i,length in enumerate(lengths):
                data = datas[i][:length]
                true_label = true_labels[i][:length]
                pred_label = pred_labels[i][:length]
                
                true_s = get_split_sentence(data,true_label,Id2Word,Id2Tag)
                pred_s = get_split_sentence(data,pred_label,Id2Word,Id2Tag)
            
                right_num = get_right_count(true_label,pred_label,Id2Tag)
                # right_s = [i for i in pred_s if i in true_s]
                TP += right_num
                FP += len(pred_s) - right_num
                FN += len(true_s) - right_num
            
            #if len(right_s) != len(pred_s) :
            #    print_result(Id2Word,Id2Tag,data,label,pred_label)
        
        precision = float(TP)/(TP+FP+0.00001)
        recall = float(TP)/(TP+FN+0.00001)
        F1 = 2*precision*recall/(precision+recall+0.00001)
        #print("epoch:{%d} precision:{%.2f}%%,recall:{%.2f}%%,F1:{%.2f}%%"%(
        #            epoch,precision*100.0,recall*100.0,F1*100.0))
        print("test | precision : %.2f%% , recall : %.2f%% , F1 : %.2f%%"%(
                    precision*100.0,recall*100.0,F1*100.0))
        

def trysplit(s):
    model.eval()
    lens = torch.tensor([len(s)])
    data = [Vocab[i] if i in Vocab else Vocab[VVO] for i in s]
    data = torch.tensor(data).long().reshape(1,-1)
    data = data.to(DEVICE)
    score,path = model(data,lens)
    pred_label = path[0].cpu().numpy()
    data = data.cpu().numpy()[0]
    pred_result = get_split_sentence(data,pred_label,Id2Word,Id2Tag)
    
    print("pred result : ",' '.join(pred_result))
    print("pred label : ",' '.join([Id2Tag[i] for i in pred_label]))
    
if __name__ == "__main__":
    if args.train:
        if args.model and os.path.exists(args.model):
            print("load model...")
            pth = torch.load(args.model)
            model.load_state_dict(pth['model'])
            optimizer.load_state_dict(pth['optimizer'])
            print('load finished')
            
        for epoch in range(args.epoch):
            print('************epoch:%d*************'%(epoch))
            train()
            test()
            
            save_path  = './model/model.pth'
            if args.model :
                save_path = args.model
                
            torch.save({"model":model.state_dict(),
                "optimizer":optimizer.state_dict()},
               save_path)
        
    else:
        if args.predict_model:
            print("load model...")
            pth = torch.load(args.predict_model)
            model.load_state_dict(pth['model'])
            optimizer.load_state_dict(pth['optimizer'])
            print('load finished')
        
        if args.test:
            test()
            sys.exit()
        
        while True:
            st = input("请输入一行中文句子（quit退出）：")
            if st == 'quit':
                break
            trysplit(st)
        