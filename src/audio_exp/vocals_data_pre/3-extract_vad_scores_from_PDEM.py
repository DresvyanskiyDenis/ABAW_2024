import audeer
import audonnx
import numpy as np
import pandas as pd
import audinterface
import os


def rename_file(old_path, new_name):
    try:
        directory, old_filename = os.path.split(old_path)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"File name successfully changed:{old_path} -> {new_path}")
    except OSError as e:
        print(f"Wrong with changing file name:{e}")


# Below is a github repo link of PDEM for your reference in which you can find the documentation for using PDEM. 
# Link：https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

# This is the PDEM model url. 
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
model_cache_root = audeer.mkdir('/media/legalalien/Data1/Androids-Corpus/data_pre/pdem_cache')
model_root = audeer.mkdir('/media/legalalien/Data1/Androids-Corpus/data_pre/pdem_model')

# Step 1: Download the PDEM model to the model_root dir and Load the PDEM model
# archive_path = audeer.download_url(url, model_cache_root, verbose=True)
# audeer.extract_archive(archive_path, model_root)

# Load model from model_root using audonnx package.
model = audonnx.load(model_root)
# Set audio sampling rate. 
sampling_rate = 16000

# Define model interface, output is valence, arousal and donimance scores. 
logits = audinterface.Feature(
    model.labels('logits'),
    process_func=model,
    process_func_args={
        'outputs': 'logits',
    },
    sampling_rate=sampling_rate,
    resample=True,
    num_workers=8, 
    verbose=True,
)


# Step 2: Indicate the train, dev and test index files respectively which already got from scripts 1-3.

data_root = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index_sort_filter.csv'

# Load index file
data_pd = pd.read_csv(data_root) # Load the index file

data_pd.set_index('Segment Path', inplace=True) # Set the index to 'Audio_clip_path' column
data_pd.index = data_pd.index.astype(str) # Convert the index to string data type.


# Set the output files' (i.e. VAD scores' files) root directory
cache_root = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/PDEM_outputs'

# Step 3: Extracting VAD scores using PDEM model for each wav (indicated by the Daic_woz_dev.index loading from the index file).
dev_features_w2v2 = logits.process_index(
    data_pd.index,
    root='',
    # The output files will save to the below cache_root directory. The file name will be a random number ending with '.pkl'
    cache_root=audeer.path(cache_root, 'VAD'),
)
# Because the file name is random number, so here we change it to a readable name.
VAD_dir_path = os.path.join(cache_root, 'VAD')
files = os.listdir(VAD_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(VAD_dir_path, file)
        rename_file(file_path,'VAD.pkl')

# The process below is same as above.
# train_features_w2v2 = logits.process_index(
#     Daic_woz_train.index,
#     root='',
#     cache_root=audeer.path(cache_root, 'vad_train'),
# )
# files = os.listdir(cache_root)
# for file in files:
#     if file.endswith('.pkl'):
#         file_path = os.path.join(cache_root, file)
# rename_file(file_path,'vad_train.pkl')


# test_features_w2v2 = logits.process_index(
#     Daic_woz_test.index,
#     root='',
#     cache_root=audeer.path(cache_root, 'vad_test'),
# )
# files = os.listdir(cache_root)
# for file in files:
#     if file.endswith('.pkl'):
#         file_path = os.path.join(cache_root, file)
# rename_file(file_path,'vad_test.pkl')
