import pretty_midi
import torch

def melody_to_numpy(fpath="ashover12.mid", unit_time=0.125):
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