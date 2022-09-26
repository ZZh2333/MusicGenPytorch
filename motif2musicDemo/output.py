import torch
import numpy as np
import pretty_midi
from mask import mask_pad, mask_tril

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# from main import predict

# 预测函数
def predict(x):
    # x = [1, 50]
    model.eval()

    # [1, 1, 50, 50]
    mask_pad_x = mask_pad(x)

    # 初始化输出,这个是固定值
    # [1, 50]
    # [[0,2,2,2...]]
    target = [131] + [130] * 801
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [1, 50] -> [1, 50, 32]
    x = model.embed_x(x)

    # 编码层计算,维度不变
    # [1, 50, 32] -> [1, 50, 32]
    x = model.encoder(x, mask_pad_x)

    # 遍历生成第1个词到第49个词
    for i in range(801):
        # [1, 50]
        y = target

        # [1, 1, 50, 50]
        mask_tril_y = mask_tril(y)

        # y编码,添加位置信息
        # [1, 50] -> [1, 50, 32]
        y = model.embed_y(y)

        # 解码层计算,维度不变
        # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,39分类
        # [1, 50, 32] -> [1, 50, 39]
        out = model.fc_out(y)

        # 取出当前词的输出
        # [1, 50, 39] -> [1, 39]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target



def numpy_to_midi(sample_roll, output='output.mid'):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    t = 0
    for i in sample_roll:
        if 'torch' in str(type(i)):
            pitch = int(i.max(0)[1])
        else:
            pitch = int(np.argmax(i))
        if pitch < 128:
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
            t += 1 / 8
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pretty_midi.Note(
                    velocity=100, pitch=int(p), start=0, end=t)
            note = pretty_midi.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + 1 / 8)
            piano.notes.append(note)
            t += 1 / 8
        elif pitch == 129:
            t += 1 / 8
    music.instruments.append(piano)
    music.write(output)

def melody_to_numpy(fpath, unit_time=0.125):
    # 首先取出midi中的音乐信息，将第一个音轨提出来，然后取所有音符元素。
    music = pretty_midi.PrettyMIDI(fpath)
    notes = music.instruments[0].notes

    # 纪录一个时间戳 t和保存向量的列表 roll。
    t = 0.
    roll = list()
    # print(notes[0], notes[-1])

    # 在 Notes里遍历，找出所有音符的音高和长度，同时不放过休止符。
    for note in notes:
        # print(t, note)

        # 两个相邻的音符不是无缝连接，说明休止符存在。计算其相对于最小分辨率 unit_time的相对时长 T，建立一个(T, 130)的矩阵，将第129维置1.
        elapsed_time = note.start - t
        if elapsed_time > 0.:
            steps = torch.zeros((int(round(elapsed_time / unit_time)), 130))
            steps[range(int(round(elapsed_time / unit_time))), 129] += 1.
            roll.append(steps)

        # 如果是无缝连接，那么检查当前音符：
        n_units = int(round((note.end - note.start) / unit_time))
        steps = torch.zeros((n_units, 130))
        steps[0, note.pitch] += 1
        steps[range(1, n_units), 128] += 1

        # 其中除第一列记录pitch外，其他列都记录sustain的128.最后合成为一个矩阵：
        roll.append(steps)
        t = note.end
    return torch.cat(roll, 0)


def array_to_single(array):
    midi_batch = []
    for i in array:
        num = 0
        for j in i:
            if j == 1:
                midi_batch.append(num)
                num = 0
                break
            num += 1
    return midi_batch


music_path = '.\MTD0760.mid'
music_batch = melody_to_numpy(music_path)
music_array = array_to_single(music_batch)


model = torch.load('.\motifToMusic.pth')

# [firstmidi,lenmidi,high_lowmidi,midiFrequence,midmidi]

x = [131] + music_array + [132] + [130] * (800 - len(music_array))
# print(x)
x = torch.LongTensor(x)

# print(''.join([" "+ str(i) for i in predict(x.unsqueeze(0))[0].tolist()]))
y = ''.join([" "+ str(i) for i in predict(x.unsqueeze(0))[0].tolist()])
# print(y)

midi_batch = []

for i,mus in enumerate(y.split(" ")):
    if(mus == str(132)):
        break
    if(mus != str(131) and mus != ""):
        midi_batch.append(mus)
print(midi_batch)

midi_array = []
for i in midi_batch:
    z = [0] * 130
    z[int(i)] = 1
    midi_array.append(z)

# print(torch.LongTensor(midi_array))
numpy_to_midi(torch.LongTensor(midi_array),'./Music.mid')