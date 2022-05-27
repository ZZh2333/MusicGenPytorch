from modulefinder import Module
from pickletools import optimize
import re
from turtle import forward
from cv2 import transpose
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1,2)))
        output = self.conv2(output).transpose(1,2)
        return self.layer_norm(output+residual)


# 3.位置编码函数
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 位置编码的实现对应正余弦公式，当然也可以自定义位置编码方式
        # 需要注意的是偶数与奇数在公式上有一个共同的部分，可以使用 log函数把次方拿下来，方便计算
        # 假设 d_model为 512，那么公式里的 pos代表的从 0，1，2，3...511的每一个位置，2i对应的取值就是 0，2，4...510，i的值为 0到255

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))    # 奇偶项公有部分中的一部分
        pe[:,0::2] = torch.sin(position * div_term) # pe[:,0::2]表示从0开始到最后，补长为 2，即所有的偶数位置
        pe[:,1::2] = torch.cos(position * div_term) # pe[:,1::2]表示从1开始到最后，补长为 2，即所有的奇数位置
        # 完成后获得 pe:[max_len * d_model]

        # 下一步代码完成后得到的形状为 [max_len * 1 * d_model]
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)  # 规定一个缓冲区（可以理解为告诉模型这个参数不进行更新操作，为常规参数）

    def forward(self, x):
        # print(x.size())

        # x:[seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)


# 4.get_attn_pad_mask
# 例如：当前句子长度为5，在后面注意力机制的部分，在计算出QK转置除以根号dk之后，softmax之前，我们得到的形状为 [len_input, len_input]
# 代表每个单词对其余包含自身的单词的影响力（相关性）
# 用另一个相同大小的符号矩阵告诉模型哪个位置是PAD填充的部分，在之后的计算中会把该位置置为无穷小
# 这里得到的矩阵形状为 [batch_size, len_q, len_k]，我们对 K中的 pad符号进行标识，并没有对 Q中做标识，因为没必要
# seq_q和 seq_k不一定一致（自注意力层一致，交互注意力层不一定一致）
# 在交互注意力中，Q来自解码端，K来自编码端，所以告诉模型编码这边 pad符号信息即可，解码端的 pad信息在交互注意力层是没有用到的
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # [batch_size, 1, len_k],1 is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)   # [batch_size, len_q, len_k]

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape),k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# 7.ScaledDotProductAttention，计算Attention公式： softmax（（Q * K转置）/根号下dk） * V
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()

    def forward(self, Q, K, V ,attn_mask):
        # 输入的形状为 Q:[batch_size, n_heads, len_q, d_k] K:[batch_size, n_heads, len_k, d_k] V:[batch_size, n_heads, len_k, d_v]
        # 经过 matmul函数得到的 scores形状为：[batch_size, n_heads,len_q, len_k]
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)

        # print(scores.size())
        # 关键点：把被 mask的地方置为无穷小，softmax之后便为0，来达到 pad的地方对 q的单词不起作用的效果
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)   # 横行做 softmax运算
        context = torch.matmul(attn, V)
        return context, attn


# 6.多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        # 输入进来的 QKV矩阵是相等的，我们会使用 Linear做映射得到参数矩阵 Wq、Wk、Wv
        # 需要保证 QK矩阵的维度相同
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K ,V, attn_mask):
        # 多头注意力机制：首先映射分头，然后计算 atten_scores，然后计算 atten_value
        # 输入进来的数据形状：Q:[batch_size, len_q, d_model], K:[batch_size, len_k, d_model], V:[batch_size, len_k, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B,S,D) -proj-> (B,S,D) -split-> (B,S,H,W) -trans-> (B,H,S,W)

        # print(residual.size())
        # 先映射，后分头，注意 q和 k分头之后维度是一致的，所以都为 d_k
        q_S = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_S = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_S = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        # print(q_S.size())
        # print(k_S.size())
        # print(v_S.size())
        # 输入进行的 attn_mask形状是：[batch_size, len_q, len_k]，然后经过下面的代码得到新的 attn_mask：[batch_size, n_heads, len_q, len_k]，即把 pad信息重复到了 n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算Attention公式： softmax（（Q * K转置）/根号下dk） * V
        context, attn = ScaledDotProductAttention()(q_S, k_S, v_S, attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn




# 5.EncoderLayer：包含两个部分，1.多头注意力机制 2.前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn = MultiHeadAttention()   # 自注意力层（多头注意力机制）
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈神经网络层（Linear层）

    def forward(self,enc_inputs,enc_self_attn_mask):
        # 自注意力层，输入是 enc_inputs，形状是 [batch_size, seq_len_q, d_model]
        # 需要注意的是最初的 QKV矩阵是等同于这个输入的
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs的形状为 [batch_size, len_q, d_model]
        return enc_outputs,attn


# 10.DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # print(dec_inputs.size())
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask) # 自注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask) # 交互注意力层
        dec_outputs = self.pos_ffn(dec_outputs) # 前馈神经网络

        return dec_outputs, dec_self_attn, dec_enc_attn

# 2.Encoder编码器包含三个部分：词向量Embedding、位置编码部分、注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size,d_model) # 定义生成一个（词表到 d_model）矩阵，大小为 [src_vocab_size , d_model]
        self.pos_emb = PositionalEncoding(d_model)  # 位置编码函数，通常为固定的正余弦函数，也可以使用类似词向量的 nn.Embedding获得一个可以更新学习的位置编码
        # 使用 ModuleList对多个 encoder进行堆叠，因为后续的 encoder并没有使用词向量和位置编码，所以抽离出来
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # 编码器接收一个输入
    def forward(self,enc_inputs):
        # enc_inputs的形状为 [batch_size, source_len]

        # print(enc_inputs.size())

        # 通过 src_emb进行索引定位， enc_outputs输出形状是 [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        # print(enc_outputs.size())

        # 位置编码函数，把两者相加后放入到这个函数中
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)
        # print(enc_outputs.size())


        # get_attn_pad_mask作用：通过符号矩阵告诉后面的模型层在原始句子中的输入中哪些部分是被 pad符号填充的
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs,enc_self_attn = layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs,enc_self_attn


# 9.解码器
# 解码端自注意力层有两处 mask操作：1、pad符号的 mask   2、当前输入之后的 mask
# 解码端交互注意力层只对编码曾 pad部分做mask
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)    #词向量
        self.pos_emb = PositionalEncoding(d_model)  #位置编码
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(n_layers))    # decoder堆叠

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs:[batch_size, target_len]
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1)

        # get_attn_pad_mask 自注意力层的时候的 pad部分，得到 pad符号矩阵
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # get_attn_subsequent_mask 做自注意力层的 mask部分，即当前单词往后的内容看不到，使用一个上三角为 1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两个矩阵相加，大于 0的部分为 1，不大于 0的部分为 0，为 1的部分在之后的操作中会被 fill到无穷小
        # 最终得到仍为一个符号矩阵，为 1的部分便是被 mask的部分
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask),0)

        # 交互注意力层的 mask符号矩阵，enc的输入为 k，便去看这个 k里面哪些是 pad符号，给到后面的模型（q也有 pad符号，但模型不在意，不进行处理）
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# 1.从整体网络来看，transformer分为三个部分：编码层、解码层、输出层
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.encoder = Encoder() # 编码曾
        self.decoder = Decoder() # 解码层
        # 输出哦快：将 embedding的维度映射到解码端词表的大小，之后做 softmax根据词的概率进行预测
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias = False)

    # 对于整个 Transformer有两个输入，编码端的输入与解码端的输入
    def forward(self,enc_inputs,dec_inputs):
        # enc_inputs的形状为 [batch_size, src_len]，dec_inputs的形状为[batch_size,tgt_len]

        # enc_inputs作为编码端的输入，输出由自己的函数内部指定，可以是全部 tokens的输出，可以是特定每一层的输出，也可以是中间某些参数的输出
        # enc_outputs就是主要的输出，enc_self_attns为QK转置相乘之后进行 softmax操作的矩阵值，代表的每个单词和其他单词的相关性，（可视化操作）
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs是 decoder的主要输出，用于后续的 Linear映射，dec_self_attns类比于 enc_self_attns 是查看每个单词对 decoder中输入的其余单词的相关性
        # dec_enc_attns是 decoder中每个单词对 encoder中每个单词的相关性
        # enc_inputs的作用：todo
        # print(enc_outputs.size())

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outpus做映射到词表的大小
        # print(dec_outputs.size())
        dec_logits = self.projection(dec_outputs)
        # print(dec_logits.size())

        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

if __name__ == "__main__":
    # 句子的输入部分
    # S表示Start，E表示End，P为填充字符（Pad，后续中值为负无穷）
    # 对应[编码端的输入，解码端的输入，真实标签]
    # Teacher forcing操作：由于在解码端的操作是无法并行的（按照时间步长依次预测），因此在训练中可将真实标签作为解码端的输入
    sentences = ['ich mochte sin bier P', 'S i want a bear', 'i want a bear E']

    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2,'sin': 3,'bier':4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P':0, 'i': 1,'want': 2,'a': 3,'bear': 4,'S': 5,'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # 编码端的输入长度
    tgt_len = 5 # 解码端的输入长度

    # 模型参数
    d_model = 512 # 每一个字符转换为 Embedding时的大小
    d_ff = 2048 # 前馈神经网络的 Linear层映射到多少维度
    d_k = d_v = 64
    n_layers = 6 # Encoder块的数量
    n_heads = 8 # 多头注意力机制的数量

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # enc_inputs = torch.LongTensor([[28,251,82,63,144]])
    # dec_inputs = torch.LongTensor([[348,576,576,576,336]])
    # target_batch = torch.LongTensor([[348,576,576,576,336]])



    # print(target_batch.contiguous().view(-1))

    for epoch in range(2):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # print(outputs.size())
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', "%04d"%(epoch+1),'cost=','{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # print('first head of last state enc_self_attns')
    # showgraph(enc_self_attns)

    # print('first head of last state dec_self_attns')
    # showgraph(dec_self_attns)

    # print('first head of last state dec_enc_attns')
    # showgraph(dec_enc_attns)