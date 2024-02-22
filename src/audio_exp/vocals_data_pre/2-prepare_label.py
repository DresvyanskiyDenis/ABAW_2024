import pandas as pd
import os
import numpy as np

def prepare_labels(csv_file, annotation_dir, train_index_file, validation_index_file, fps=30):
    # 读取csv文件
    df = pd.read_csv(csv_file)

    # 新增两列用于存放valence和arousal的均值
    df['Valence'] = np.nan
    df['Arousal'] = np.nan

    # 创建两个DataFrame用于存放训练集和验证集的数据
    df_train = pd.DataFrame(columns=df.columns)
    df_validation = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        # 根据Original Audio列的名称，找到对应的annotation文件
        annotation_file_train = os.path.join(annotation_dir, 'Train_Set', row['Original Audio'] + '.txt')
        annotation_file_validation = os.path.join(annotation_dir, 'Validation_Set', row['Original Audio'] + '.txt')

        if os.path.exists(annotation_file_train):
            annotation_file = annotation_file_train
            df_target = df_train
        elif os.path.exists(annotation_file_validation):
            annotation_file = annotation_file_validation
            df_target = df_validation
        else:
            continue

        # 读取annotation文件
        df_annotation = pd.read_csv(annotation_file)

        # 根据Start Time和End Time的值转换为FPS的对应数值，然后转换为对应的行
        start_frame = int(row['Start Time'] * fps)
        end_frame = int(row['End Time'] * fps)

        # # 对其中的valence和arousal计算其均值
        # row['Valence'] = df_annotation['valence'][start_frame:end_frame].mean()
        # row['Arousal'] = df_annotation['arousal'][start_frame:end_frame].mean()
        
        # 对其中的valence和arousal计算其均值
        valence = df_annotation['valence'][start_frame:end_frame]
        arousal = df_annotation['arousal'][start_frame:end_frame]

        # 去除小于-1的和大于1的值
        valence = valence[(valence >= -1) & (valence <= 1)]
        arousal = arousal[(arousal >= -1) & (arousal <= 1)]

        # 计算mean值
        row['Valence'] = valence.mean()
        row['Arousal'] = arousal.mean()

        # 将行添加到对应的DataFrame
        df_target = df_target.append(row, ignore_index=True)

        # 根据df_target更新df_train或df_validation
        if annotation_file == annotation_file_train:
            df_train = df_target
        else:
            df_validation = df_target

    # 将新的annotaion写到两个新的index文件中
    df_train.to_csv(train_index_file, index=False)
    df_validation.to_csv(validation_index_file, index=False)


# 使用示例

input_csv_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index_sort_filter.csv'
annotation_dir = '/media/legalalien/Data1/ABAW_6th/6th_ABAW_Annotations/VA_Estimation_Challenge'
train_index_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/train_index.csv'
validation_index_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/validation_index.csv'
prepare_labels(input_csv_file, annotation_dir, train_index_file, validation_index_file, fps=30)
