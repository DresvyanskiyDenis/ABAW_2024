import pandas as pd

# 读取csv文件
df = pd.read_csv('/media/legalalien/Data1/ABAW_6th/vocals_data_pre/wav_path_dur.csv')

# 根据Audio_id进行分组，并计算每组的Duration总和
total_duration = df.groupby('Audio_id')['Duration'].sum()

# 打印结果
print(total_duration["1-30-1280x720"])