import os
from typing import Optional

import pandas as pd
import torch
import pandas

import cv2
import numpy as np
from PIL import Image

from src.video.preprocessing.utils import extract_face_according_bbox, recognize_faces_bboxes, \
    get_bbox_closest_to_previous_bbox, get_most_confident_person, load_and_prepare_detector_retinaFace_mobileNet


def extract_face_one_video(path_to_video:str, detector:object, output_path:str, keep_the_same_person:Optional[bool]=False,
                                 every_n_frame:Optional[int]=1)->pd.DataFrame: # TODO: check it
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
            if counter-1 % every_n_frame == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the face
                bboxes = recognize_faces_bboxes(frame, detector, conf_threshold=0.8)
                # create new filename for saving the face
                new_filename = os.path.basename(path_to_video).split(".")[0]
                new_filename = os.path.join(new_filename,
                                            f"{counter:05}.jpg")
                # if not recognized, note it as NaN
                if bboxes is None:
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
            # increment counter
            counter += 1
            # save previous face and bbox
            previous_face = face
            last_bbox = bbox
        else:
            break
    return metadata



if __name__=="__main__":
    path_to_video = r"F:\Datasets\AffWild2\videos\2-30-640x360.mp4"
    path_to_output = r"F:\Datasets\AffWild2\preprocessed"
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output, exist_ok=True)
    detector = load_and_prepare_detector_retinaFace_mobileNet()
    metadata = extract_face_one_video(path_to_video, detector, path_to_output, keep_the_same_person=True, every_n_frame=1)