import os
from typing import Optional

import pandas as pd
import torch
import pandas

import cv2
import numpy as np
from PIL import Image

from src.video.preprocessing.utils import recognize_one_face_bbox, extract_face_according_bbox


def extract_face_for_every_frame(path_to_video:str, detector:object, output_path:str, keep_the_same_person:Optional[bool]=False,
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
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter % every_n_frame == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the face
                bbox = recognize_one_face_bbox(frame, detector)
                # create new filename for saving the face
                new_filename = os.path.basename(path_to_video).split(".")[0]
                new_filename = os.path.join(new_filename,
                                            f"{counter:05}.jpg")
                # if not recognized, note it as NaN
                if bbox is None:
                    # save previous face
                    face = previous_face
                else:
                    # otherwise, extract the face and save it
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
            previous_face = face
        else:
            break
    return metadata
