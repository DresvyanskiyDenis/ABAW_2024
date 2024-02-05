"""
This is the script for extracting audio from video with/without filtering speech.
"""

import os
import wave
import shutil
import subprocess

from tqdm import tqdm


def get_duration(file_path: str) -> float:
    """Return duration of the audio file

    Args:
        file_path (str): The path to the file

    Returns:
        float: Duration of the file in seconds
    """
    duration_seconds = 0
    with wave.open(file_path) as wav:
        duration_seconds = wav.getnframes() / wav.getframerate()

    return duration_seconds


def convert_without_filtering(inp_path: str, 
                              out_path: str, 
                              checking: bool = True) -> None:
    """Convert video to audio using ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the ffmpeg command. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # sample rate 16000
    command = f"ffmpeg -y -i {inp_path} -vn -acodec pcm_s16le -ar 16000 {out_path}"
       
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)


def convert_with_filtering(inp_path: str, 
                           out_path: str, 
                           checking: bool = True) -> None:
    """Extract speech from the video file using Spleeter and ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # 44100 for spleeter
    command = f"ffmpeg -y -i {inp_path} -vn -acodec pcm_s16le -ar 44100 {out_path}"
       
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    inp_duration = get_duration(out_path)

    # extract speech using spleeter
    command = f"spleeter separate -o {out_dirname} {out_path} -d 1620"
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, env=os.environ.copy()
        )

    spleeter_duration = get_duration(out_path)

    # convert 44100 to 16000
    command = "ffmpeg -y -i {0} -ar 16000 -ac 1 {1}".format(
        os.path.join(
            out_dirname, os.path.basename(out_path).split(".")[0], "vocals.wav"
        ),
        out_path,
    )
    
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        
        shutil.rmtree(
            os.path.join(out_dirname, os.path.basename(out_path).split(".")[0])
        )

    # check results for errors
    final_duration = get_duration(out_path)
    if (abs(inp_duration - spleeter_duration) < 1e-4) and (abs(inp_duration - final_duration) < 1e-4):
        pass
    else:
        print(f"Error {inp_path}")
        print(inp_duration, spleeter_duration, final_duration)


def convert_video_to_audio(files_root: str, 
                           wavs_root: str = 'wavs',
                           vocals_root: str = 'vocals',
                           filtering: bool = False,
                           checking: bool = True) -> None:
    """Loop through the directory, and extract speech from each video file using Spleeter and ffmpeg.

    Args:
        files_root (str): Input directory
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    # run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    out_wavs_root = os.path.join(os.path.dirname(files_root), wavs_root)
    out_vocals_root = os.path.join(os.path.dirname(files_root), vocals_root)

    for dn, _, fns in os.walk(os.path.join(files_root)):
        if not fns:
            continue

        for fn in tqdm(fns):
            convert_without_filtering(
                inp_path=os.path.join(dn, fn),
                out_path=os.path.join(
                    dn.replace(files_root, out_wavs_root),
                    fn.replace("mp4", "wav").replace("avi", "wav"),
                ),
                checking=checking,
            )
            
            if filtering:
                convert_with_filtering(
                    inp_path=os.path.join(dn, fn),
                    out_path=os.path.join(
                        dn.replace(files_root, out_vocals_root),
                        fn.replace("mp4", "wav").replace("avi", "wav"),
                    ),
                    checking=checking,
                )


if __name__ == "__main__":
    files_root = "/media/maxim/Databases/ABAW2024/data/videos"
    
    convert_video_to_audio(files_root=files_root, 
                           filtering=False,
                           checking=False)
