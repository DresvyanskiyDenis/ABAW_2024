import pandas as pd
import os
import numpy as np

def prepare_labels(csv_file, annotation_dir, train_index_file, validation_index_file, fps=30):
    # Read the csv file
    df = pd.read_csv(csv_file)

    # Add two new columns to store the mean of valence and arousal
    df['Valence'] = np.nan
    df['Arousal'] = np.nan

    # Create two DataFrames to store the data for the training set and validation set
    df_train = pd.DataFrame(columns=df.columns)
    df_validation = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        # Find the corresponding annotation file based on the name in the 'Original Audio' column
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

        # Read the annotation file
        df_annotation = pd.read_csv(annotation_file)

        # Convert the values of 'Start Time' and 'End Time' to corresponding frame numbers based on FPS, then convert to corresponding rows
        start_frame = int(row['Start Time'] * fps)
        end_frame = int(row['End Time'] * fps)

        # Calculate the mean of valence and arousal
        valence = df_annotation['valence'][start_frame:end_frame]
        arousal = df_annotation['arousal'][start_frame:end_frame]

        # Remove values less than -1 and greater than 1
        valence = valence[(valence >= -1) & (valence <= 1)]
        arousal = arousal[(arousal >= -1) & (arousal <= 1)]

        # Calculate the mean value
        row['Valence'] = valence.mean()
        row['Arousal'] = arousal.mean()

        # Add the row to the corresponding DataFrame
        df_target = df_target.append(row, ignore_index=True)

        # Update df_train or df_validation based on df_target
        if annotation_file == annotation_file_train:
            df_train = df_target
        else:
            df_validation = df_target

    # Write the new annotation to two new index files
    df_train.to_csv(train_index_file, index=False)
    df_validation.to_csv(validation_index_file, index=False)


# Usage example

input_csv_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index_sort_filter.csv'
annotation_dir = '/media/legalalien/Data1/ABAW_6th/6th_ABAW_Annotations/VA_Estimation_Challenge'
train_index_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/train_index.csv'
validation_index_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/validation_index.csv'
prepare_labels(input_csv_file, annotation_dir, train_index_file, validation_index_file, fps=30)