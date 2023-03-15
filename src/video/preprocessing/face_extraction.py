import glob
import os
from typing import Optional, List

import pandas as pd
import torch
import pandas

import cv2
import numpy as np
from PIL import Image
from moviepy.video.fx import crop
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.video.preprocessing.face_extraction_utils import extract_face_according_bbox, recognize_faces_bboxes, \
    get_bbox_closest_to_previous_bbox, get_most_confident_person, load_and_prepare_detector_retinaFace_mobileNet


def extract_face_one_video(path_to_video:str, detector:object, output_path:str, keep_the_same_person:Optional[bool]=False,
                                 every_n_frame:Optional[int]=1)->pd.DataFrame:
    """ Extracts faces from video and saves them to the output path. Also, generates dataframe with information about the
    extracted faces.

    :param path_to_video: str
            path to the video file
    :param detector: object
            the model that generates bboxes from facial images. It should return bounding boxes and confidence score for every person.
    :param output_path: str
            path to the output directory
    :param keep_the_same_person: bool
            if True, then the function will try to keep the same person along the frames bu comparins positions of the
            bounding boxes.
    :param every_n_frame: int
            extract faces from every n-th frame
    :return: pd.DataFrame
            dataframe with information about the extracted faces
    """
    video_filename = os.path.basename(path_to_video).split(".")[0]
    # create output directory if needed
    if not os.path.exists(os.path.join(output_path, video_filename)):
        os.makedirs(os.path.join(output_path, video_filename), exist_ok=True)
    # metadata
    metadata = pd.DataFrame(columns=["filename", "frame_num", "timestamp"])
    # load video file
    video = cv2.VideoCapture(path_to_video)
    # get FPS
    FPS = video.get(cv2.CAP_PROP_FPS)
    FPS_in_seconds = 1. / FPS
    # go through all frames
    counter = 1
    previous_face = None
    last_bbox = None
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if (counter-1) % every_n_frame == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the face
                bboxes = recognize_faces_bboxes(frame, detector, conf_threshold=0.8)
                # create new filename for saving the face
                new_filename = video_filename
                new_filename = os.path.join(new_filename,
                                            f"{counter:05}.jpg")
                # if not recognized, note it as NaN
                if bboxes is None:
                    # if there were no face vefore
                    if previous_face is None:
                        face = np.zeros((224,224,3), dtype=np.uint8)
                        bbox = None
                    else:
                        # save previous face
                        face = previous_face
                        bbox = last_bbox
                else:
                    # otherwise, extract the face and save it
                    if keep_the_same_person and last_bbox is not None:
                        # if we want to keep the same person, then we need to calculate the distances between the
                        # center of the previous bbox and the current ones. Then, choose the closest one.
                        bbox = get_bbox_closest_to_previous_bbox(bboxes, last_bbox)
                    else:
                        # otherwise, take the most confident one
                        bbox = get_most_confident_person(bboxes)
                    # extract face according to bbox
                    face = extract_face_according_bbox(frame, bbox)
                # create full path to the output file
                output_filename = os.path.join(output_path, new_filename)
                # save extracted face
                Image.fromarray(face).save(output_filename)
                metadata = pd.concat([metadata,
                                      pd.DataFrame.from_records([{
                                          "filename": output_filename,
                                          "frame_num": counter,
                                          "timestamp": timestamp}
                                      ])
                                      ], ignore_index=True)
                # save previous face and bbox
                previous_face = face
                last_bbox = bbox
            # increment counter
            counter += 1
        else:
            break
    return metadata


def extract_faces_from_all_videos(paths_to_videos:List[str], detector:object, output_path:str, keep_the_same_person:Optional[bool]=False,
                                  every_n_frame:Optional[int]=1)->pd.DataFrame:
    """ Extracts faces from all videos and saves them to the output path. Also, generates dataframe with information about every video.

    :param paths_to_videos: List[str]
            list of paths to the video files
    :param detector: object
            the model that generates bboxes from facial images. It should return bounding boxes and confidence score for every person.
    :param output_path: str
            path to the output directory
    :param keep_the_same_person: Optional[bool]
            if True, then the function will try to keep the same person along the frames by comparing the position of the
            bounding boxes.
    :param every_n_frame: int
            extract faces from every n-th frame
    :return: pd.DataFrame
            dataframe with information about the extracted faces (all videos are included)
    """
    # create output directory if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create metadata file
    metadata = pd.DataFrame(columns=["filename", "frame_num", "timestamp"])
    # go through all videos
    for path_to_video in paths_to_videos:
        # extract faces from one video
        metadata_one_video = extract_face_one_video(path_to_video, detector, output_path, keep_the_same_person, every_n_frame)
        # add metadata from one video to the main metadata
        metadata = pd.concat([metadata, metadata_one_video], ignore_index=True, axis=0)
        # print
        print(f"Finished processing {path_to_video}")
        # save metadata
        metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False)
    return metadata

def make_left_right_videos(path_to_data:str, output_path:str)->None:
    # ration for train/dev/test videos to separate two different people
    dividing_ratios = {
            "46-30-484x360": 12.5/25.,
            "6-30-1920x1080": 14./25.,
            "video2":-1, # -1 means full frame for right and left
            "video5":12.5/25.,
            "video10_1":12.5/25.,
            "video29":11.5/25.,
            "video49": 10.5/25.,
            "video55":12./25.,
            "video59":12./25.,
            "video74":12.5/25.,
            "49-30-1280x720": 9.5/25.,
            "135-24-1920x1080": 13.5/25.,
            "130-25-1280x720":15.5/25.,
    }
    # get names of all videos with their extentions
    all_videos = glob.glob(os.path.join(path_to_data, "*"))
    all_videos = [os.path.basename(video) for video in all_videos]
    # get dict filename->extention
    all_videos = {os.path.splitext(video)[0]:os.path.splitext(video)[1] for video in all_videos}
    # create output directory if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # extract faces from all videos
    for filename, ratio in dividing_ratios.items():
        # find idx of the video in all_videos list
        full_path = os.path.join(path_to_data, filename+all_videos[filename])
        # load video file using MoviePy
        video = VideoFileClip(full_path)
        # get height and width of the video
        width, height = video.size
        # get fps
        fps = video.fps
        # form new output names
        output_name_left = os.path.join(output_path, filename+"_left"+all_videos[filename])
        output_name_right = os.path.join(output_path, filename+"_right"+all_videos[filename])
        # crop the video using ratio and Moviepy, then save it
        if ratio == -1:
            # if ratio=-1, then we need to save the full frame for both left and right videos
            video_left = video
            video_right = video
        else:
            # otherwise, we need to crop the video
            video_left = crop.crop(video, x1=0, x2=width*ratio, y1=0, y2=height)
            video_right = crop.crop(video, x1=width*ratio, x2=width, y1=0, y2=height)
        # save videos
        video_left.write_videofile(output_name_left, fps=fps)
        video_right.write_videofile(output_name_right, fps=fps)













if __name__=="__main__":
    path_to_data = r"F:\Datasets\AffWild2\videos"
    output_path = r"F:\Datasets\AffWild2\preprocessed\faces"
    # load detector
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    paths_to_videos = glob.glob(os.path.join(path_to_data, "*"))
    metadata = extract_faces_from_all_videos(paths_to_videos=paths_to_videos, detector = detector, output_path=output_path,
                                  keep_the_same_person= True, every_n_frame= 1)
    # save metadata
    metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False)


    # for creating the _left and _right videos
    """
    path_to_data = r"F:\Datasets\AffWild2\videos"
    output_path = r"F:\Datasets\AffWild2\preprocessed\left_right_videos
    make_left_right_videos(path_to_data, output_path)"""

