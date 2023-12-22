# %% [markdown]
# # 基于Transformer模型的情感分析与情感识别研究
# ## 项目简介
# 本项目通过基于Transformer模型的情感分析和情感识别研究，为社会提供了更准确、高效的情感分析技术。这将在市场调研、社交媒体分析、舆情监测等领域发挥重要作用，帮助企业了解用户情感倾向、产品满意度以及品牌声誉等关键信息，从而支持决策制定和运营优化。
# 
# 同时，对于刚接触神经网络的我来说，这个项目提供了宝贵的机会，能够通过深入研究和实践Transformer模型，提升自然语言处理和深度学习的技能，展示出扎实的研究能力和解决实际问题的能力，并为学术和职业发展奠定坚实基础。

# %% [markdown]
# ## 导入库

# %%
import torch
import copy
import collections
import os
import random
import torch.nn.functional as F
import numpy as np


from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchtext.vocab import vocab
from tqdm import tqdm

# %% [markdown]
# ## 参数设置

# %%
class Config(object):
    def __init__(self):
        self.model_name = 'Transformer'
        # 预训练词向量
        self.embedding_pretrained = None  
        # 测试设备种类
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 随机失活
        self.dropout = 0.5 
        # 类别数
        self.num_classes = 2  
        # epoch数
        self.num_epochs = 10
        # mini-batch大小  
        self.batch_size = 20  
        # 每句话处理成的长度
        self.pad_size = 500 
        # 对读取数据的部分进行赋值  
        self.n_vocab = None
        # 学习率
        self.learning_rate = 5e-4
        # 词向量维度  
        self.embed = 300  
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2
        self.checkpoint_path = './model.ckpt'

# %% [markdown]
# ## 数据预处理

# %%
class ImdbDataset(Dataset):
    def __init__(
        self, folder_path="./aclImdb", is_train=True, is_small=False
    ) -> None:
        super().__init__()
        self.data, self.labels = self.read_dataset(folder_path, is_train, is_small)

    # 读取数据
    def read_dataset(
        self,
        folder_path,
        is_train,
        small
    ):
        data, labels = [], []
        for label in ("pos", "neg"):
            folder_name = os.path.join(
                folder_path, "train" if is_train else "test", label
            )
            for file in tqdm(os.listdir(folder_name)):
                with open(os.path.join(folder_name, file), "rb") as f:
                    text = f.read().decode("utf-8").replace("\n", "").lower()
                    data.append(text)
                    labels.append(1 if label == "pos" else 0)
        # TODO: 未来可以使用这个步骤创建多个模型实现模型集成功能，例如随机删除数据或者加入一些额外的噪声
        # 打乱数据
        random.shuffle(data)
        random.shuffle(labels)
        return data, labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels
    

# 获取数据集的词元列表
def get_tokenized(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(" ")]
    return [tokenizer(review) for review in data]


# 获取数据集的词汇表
def get_vocab(data):
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # min_freq设置为5，过滤出现次数过少的
    vocab_freq = {"<UNK>": 0, "<PAD>": 1}
    # 添加满足词频条件的单词到词汇表，并分配索引
    for word, freq in counter.items():
        if freq >= 5:
            vocab_freq[word] = len(vocab_freq)
    # 构建词汇表对象并返回
    return vocab(vocab_freq)


# 数据预处理，将数据转换成神经网络的输入形式
def preprocess_imdb(train_data, vocab,config):
    # 将每条评论通过截断或者补0，使得长度变成500
    max_l = config.pad_size  
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [1] * (max_l - len(x))

    labels = train_data.get_labels()
    tokenized_data = get_tokenized(train_data.get_data())
    vocab_dict = vocab.get_stoi()
    features = torch.tensor(
        [pad([vocab_dict.get(word, 0) for word in words]) for words in tokenized_data]
    )
    labels = torch.tensor([label for label in labels])
    return features, labels


# 加载数据集
def load_data(config):
    train_data = ImdbDataset(folder_path="./aclImdb", is_train=True)
    test_data = ImdbDataset(folder_path="./aclImdb", is_train=False)
    vocab = get_vocab(train_data.get_data())
    train_set = TensorDataset(*preprocess_imdb(train_data, vocab,config))
    test_set = TensorDataset(*preprocess_imdb(test_data, vocab,config))
    print(f"训练集大小{train_set.__len__()}")
    print(f"测试集大小{test_set.__len__()}")
    print(f"词表中单词个数:{len(vocab)}")
    train_iter = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_iter = DataLoader(test_set, config.batch_size)
    return train_iter, test_iter, vocab


# 预先定义配置
config = Config()
#加载数据
train_data,test_data,vocabs_size = load_data(config)
#补充词表大小
config.n_vocab = len(vocabs_size) + 1

# %% [markdown]
# ## 模型训练

# %%
# 实现Transformer
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        #return out
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


#调用transformer的编码器
model = Model(config)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
#多分类的任务
criterion = nn.CrossEntropyLoss()
batch_size=config.batch_size

# %% [markdown]
# ## 模型评估

# %%
# 记录训练过程的数据
epoch_loss_values = []
metric_values = []
best_acc = 0.0
for epoch in range(config.num_epochs):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # training
    model.train()
    for i,train_idx in enumerate(tqdm(train_data)):
        features, labels = train_idx
        features = features.cuda()
        labels = labels.cuda()
        optimizer.zero_grad() 
        outputs = model(features) 
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        # get the index of the class with the highest probability
        _, train_pred = torch.max(outputs, 1) 
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data)):
            features, labels = batch
            features = features.cuda()
            labels = labels.cuda()
            outputs = model(features)
            loss = criterion(outputs, labels) 
            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += loss.item()

    print(f'Model Status: [{epoch+1:03d}/{config.num_epochs:03d}] | Train Acc: {train_acc/25000:3.5f} | Loss: {train_loss/len(train_data):3.5f} | Val Acc: {val_acc/25000:3.5f} | loss: {val_loss/len(test_data):3.5f}')
    
    epoch_loss_values.append(train_loss/len(train_data))
    metric_values.append(val_acc/25000)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), config.checkpoint_path)
        print(f'saving model with acc {best_acc/25000:.5f}')
    

# 画出训练过程中的损失曲线以及准确率曲线
import matplotlib.pyplot as plt
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [ (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [(i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig('result.png')
plt.show()


