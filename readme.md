#### 简介
使用BILSTM+CRF进行中文分词，语料来源：SIGHAN Bakeoff 2005  
[语料下载链接](http://sighan.cs.uchicago.edu/bakeoff2005/)
#### 文件说明
* data为训练数据的默认位置
* model为模型的默认保存位置
* data_loader包括对数据的处理
* crf.py为模型位置
* main.py为主函数
#### 运行说明
python main.py 
交互输入，输入一行文本，输出标签序列和分词结果。
可带若干参数，详情请使用python main.py --help 查看
python main.py --test
对测试集进行测试
python main.py --train

usage: main.py [-h] [--train] [--test] [--train_path TRAIN_PATH] [--test_path TEST_PATH] [--model MODEL]
               [--predict_model PREDICT_MODEL] [--epoch EPOCH] [--batch_size BATCH_SIZE]

参数说明

optional arguments:
  -h, --help            show this help message and exit
  --train               训练模式
  --test                测试模式
  --train_path TRAIN_PATH
                        指定训练文件路径，注：字典来源于训练文件
  --test_path TEST_PATH
                        指定测试文件路径
  --model MODEL         指定模型加载路径（用于训练，以及保存，默认保存在./model/model.pth）
  --predict_model PREDICT_MODEL
                        指定模型加载路径（用于测试）
  --epoch EPOCH         指定训练轮数
  --batch_size BATCH_SIZE
                        指定batch大小
