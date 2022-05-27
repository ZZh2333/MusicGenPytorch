import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from picCompute import image_colorfulness,contrast,HSVcompute
from utils import melody_to_numpy

# 2.make_batch
def make_batch(imagepath,musicpath):
     # 计算图片特征
    image = cv2.imread(imagepath)
    colorfulness = image_colorfulness(image)
    cg = contrast(image)
    h,s,v = HSVcompute(image)
    input_batch = [[colorfulness,cg,h,s,v]]

    # enc_inputs = torch.LongTensor(input_batch)
    # tensor([[ 28, 251,  82,  63, 144]])

    mid_array = melody_to_numpy(musicpath)
    # print(mid_array)
    midi_batch = []
    for i in mid_array:
        # print(i)
        num = 0
        for j in i:
            if j == 1:
                midi_batch.append(num)
                num = 0
                break
            num += 1
        # print(str(i).index(str(1)))
        # midi_batch.append(str(i).index(str(1)))
    # print(torch.LongTensor([midi_batch]))
    return torch.LongTensor(input_batch),torch.LongTensor([midi_batch]),torch.LongTensor([midi_batch])

# 5.get_sinusoid_encoding_table 位置编码公式
def get_sinusoid_encoding_table(n_position,d_model):
    def cal_angle(position,hid_idx):
        return position / np.power(10000,2*(hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position,hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

# 8.ScaledDotProductAttention，计算Attention公式： softmax（（Q * K转置）/根号下dk） * V
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()
    def forward(self,Q,K,V,attn_mask):
        # 输入的形状为 Q:[batch_size, n_heads, len_q, d_k] K:[batch_size, n_heads, len_k, d_k] V:[batch_size, n_heads, len_k, d_v]
        # 经过 matmul函数得到的 scores形状为：[batch_size, n_heads,len_q, len_k]
        scores = torch.matmul(Q,K.transpose(-1,-2)) / np.sqrt(d_k)

        # print(scores.size())
        # 关键点：把被 mask的地方置为无穷小，softmax之后便为0，来达到 pad的地方对 q的单词不起作用的效果
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)   # 横行做 softmax运算
        context = torch.matmul(attn, V)
        return context, attn      

# 7.多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        # 输入进来的 QKV矩阵是相等的，使用 Linear做映射得到参数矩阵 Wq、Wk、Wv
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
        # print(batch_size)
        # print(residual.size())

        # 先映射，后分头，注意 q和 k分头之后维度是一致的，所以都为 d_k
        q_S = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_S = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_S = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        # 输入进行的 attn_mask形状是：[batch_size, len_q, len_k]，然后经过下面的代码得到新的 attn_mask：[batch_size, n_heads, len_q, len_k]，即把 pad信息重复到了 n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # print(q_S.size())
        # print(k_S.size())
        # print(v_S.size())

        # 计算Attention公式： softmax（（Q * K转置）/根号下dk） * V
        context, attn = ScaledDotProductAttention()(q_S, k_S, v_S, attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn

# 9.
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

# 6.EncoderLayer：包含两个部分，1.多头注意力机制 2.前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn = MultiHeadAttention() # 自注意力层
        self.pos_ffn = PoswiseFeedForwardNet() # 前馈神经网络层
    def forward(self,enc_inputs,enc_self_attn_mask):
        # 自注意力层，输入是 enc_inputs，形状是 [batch_size, seq_len_q, d_model]
        # 需要注意的是最初的 QKV矩阵是等同于这个输入的
        enc_outputs,attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs:[batch_size * len_q * d_model]
        return enc_outputs,attn

# 10.get_attn_pad_mask
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
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

# 15.位置编码函数
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


# 4.Encoder编码器包含三个部分：词向量Embedding、位置编码部分、注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size,d_model)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True) # 位置编码函数，正余弦函数
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # 编码器接收一个输入
    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        # enc_inputs的形状为 [batch_size, source_len]
        # 通过 src_emb进行索引定位， enc_outputs输出形状是 [batch_size, src_len, d_model]
        
        # 位置编码函数，把两者相加后放入到这个函数中
        # enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        # print(enc_inputs.size())
        enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)
        
        # print(enc_outputs.size())
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)
        # print(enc_outputs.size())
        
        # get_attn_pad_mask作用：通过符号矩阵告诉后面的模型层在原始句子中的输入中哪些部分是被 pad符号填充的
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
        

# 12.DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # print(dec_inputs.size())
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn



# 13.
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

# 14.
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

# 11.解码器
# 解码端自注意力层有两处 mask操作：1、pad符号的 mask   2、当前输入之后的 mask
# 解码端交互注意力层只对编码曾 pad部分做mask
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.pos_emb = PositionalEncoding(d_model)  #位置编码
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        # dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1)
        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# 3.Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model,tgt_vocab_size,bias=False)
    def forward(self,enc_inputs,dec_inputs):
        enc_outputs,enc_self_attns = self.encoder(enc_inputs)
        # print(enc_outputs.size())
        dec_outputs,dec_self_attns,dec_enc_attns = self.decoder(dec_inputs,enc_inputs,enc_outputs)
        # print(dec_outputs.size())
        dec_logits = self.projection(dec_outputs)
        # print(dec_logits.size())
        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns

# 1.main
if __name__ == '__main__':

    musicpath = 'midiencode/test5Music2.mid'
    imagepath = 'pictest/Origin.jpg'

    src_len = 5
    tgt_len = 5 # 32个十六分音符
    src_vocab_size = 255    # 图像特征对应词表
    tgt_vocab_size = 130
    # tgt_vocab_size = 130

    # 模型参数
    d_model = 512 # 每一个字符转换为 Embedding时的大小
    d_ff = 2048 # 前馈神经网络的 Linear层映射到多少维度
    d_k = d_v = 64
    n_layers = 6 # Encoder块的数量
    n_heads = 8 # 多头注意力机制的数量

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)

    enc_inputs,dec_inputs,target_batch = make_batch(imagepath,musicpath)

    # print(enc_inputs)
    # print(dec_inputs)
    # print(target_batch.contiguous().view(-1))

    for epoch in range(5):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # print(outputs.size())
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', "%04d"%(epoch+1),'cost=','{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()


    # # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print([[n.item()] for n in predict.squeeze()])


    # print(dec_inputs.size())
    # print(enc_inputs.size())
    # torch.Size([32, 130])
    # torch.Size([1, 5])