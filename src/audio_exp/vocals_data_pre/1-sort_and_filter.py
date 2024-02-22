import os
import pandas as pd

input_csv_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index.csv'
output_csv_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index_sort_filter.csv'

# 读取csv文件
df = pd.read_csv(input_csv_file)

# 根据'Segment Duration'列的数值，筛除掉时长大于160秒，小于0.5秒的行
df_filtered = df[(df['Segment Duration'] >= 0.5) & (df['Segment Duration'] <= 160)]

# 根据'Original Audio'和'Segment Audio'列进行排序
df_sorted = df_filtered.sort_values(by=['Original Audio', 'Segment Audio'])

# 生成新的csv文件
df_sorted.to_csv(output_csv_file, index=False)



