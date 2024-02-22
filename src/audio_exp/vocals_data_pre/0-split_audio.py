import os
import librosa
import numpy as np
import soundfile as sf
import csv


def split_audio_files(input_dir, output_dir, output_index_file, top_db=70, frame_length=2048, hop_length=512):
    with open(output_index_file, 'w', newline='') as index_file:
        writer = csv.writer(index_file)
        writer.writerow(['Original Audio', 'Segment Audio', 'Segment Path', 'Start Time', 'End Time', 'Segment Duration'])

        for dirpath, dirnames, filenames in os.walk(input_dir):
            for file in filenames:
                if file.endswith('.wav') or file.endswith('.mp3'):
                    input_path = os.path.join(dirpath, file)
                    y, sr = librosa.load(input_path, sr=None)

                    intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

                    relative_path = os.path.relpath(dirpath, input_dir)
                    audio_name = os.path.splitext(file)[0]  # obatin the name of the audio file without the extension
                    segment_output_dir = os.path.join(output_dir, relative_path, audio_name)
                    os.makedirs(segment_output_dir, exist_ok=True)

                    for i, (start, end) in enumerate(intervals):
                        segment = y[start:end]
                        output_filename = f"{audio_name}_{i:03}.wav"
                        output_path = os.path.join(segment_output_dir, output_filename)
                        sf.write(output_path, segment, sr)

                        start_time = start / sr
                        end_time = end / sr
                        segment_length = librosa.get_duration(y=segment, sr=sr)
                        print(f"Segment {output_filename} length: {segment_length} seconds")

                        writer.writerow([audio_name, output_filename, output_path, start_time, end_time, segment_length])


input_dir = '/media/legalalien/Data1/ABAW_6th/vocals'
output_dir = '/media/legalalien/Data1/ABAW_6th/vocals_split'
output_index_file = '/media/legalalien/Data1/ABAW_6th/vocals_data_pre/index.csv'
split_audio_files(input_dir, output_dir, output_index_file) # generated index file, generated audio segment files