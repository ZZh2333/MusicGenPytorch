import pretty_midi
import torch
import os

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


# 计算音高范围
def computeHighLow(list):
    min = list[0]
    max = list[0]
    for i in list:
        if(i>max):
            max = i
        if(i<min):
            min = i
    return max-min

# 计算变化频率
def computeFrequence(list):
    cnt = 0
    for i,m in enumerate(list):
        if(i!=0 and m!=list[i-1] and i!=128):
            cnt += 1
    return cnt

# 计算平均音高
def computeMidMidi(list):
    sum = 0
    len = 0
    for i in list:
        if i != 128:
            sum += i
            len += 1
    return sum/len


def getMusicData():
    # musicPath = "./output_MTD_min_data/MTD6657_Prokofiev_Op064-10.mid"
    # midiarray = melody_to_numpy(musicPath)
    # midi_batch = array_to_single(midiarray)
    # print(midi_batch)

    musicData = []
    musicFeatures = []
    # 填充字符
    padding = 130
    # print("helloworld")
    # baseDir = os.path.dirname(os.path.abspath(__name__))
    # musicDataPath = str(baseDir+'/mainTransformerTest/output_MTD_min_data')
    # print(musicDataPath)
    for filepath,dirnames,filenames in os.walk(r'E:\Anaconda\project\envs\MusicGenPytorch\mainTransformerTest\output_MTD_min_data'):
        print("开始处理音频数据")
        for filename in filenames:
            try:
                # print(os.path.join(filepath,filename))
                musicpath = os.path.join(filepath,filename)
                midiarray = melody_to_numpy(musicpath)
                midi_batch = array_to_single(midiarray)
                # print(midi_batch[0])
                if(len(midi_batch)<=64):
                    firstmidi = midi_batch[0]
                    lenmidi = len(midi_batch)
                    high_lowmidi = computeHighLow(midi_batch)
                    midiFrequence = computeFrequence(midi_batch)
                    midmidi = computeMidMidi(midi_batch)
                    # midi_batch += [padding] * (64 - len(midi_batch))
                    musicData.append(midi_batch)
                    musicFeatures.append([firstmidi,lenmidi,high_lowmidi,midiFrequence,midmidi])
            except Exception as e:
                # print(e)
                continue
    # print("包含"+str(len(musicData))+"条音频数据")
    # print("cnt:"+str(cnt))
    return musicFeatures, musicData