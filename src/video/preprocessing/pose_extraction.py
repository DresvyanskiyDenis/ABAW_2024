import glob
import os
from typing import Optional, List

import pandas as pd
import torch
import pandas

import cv2
import numpy as np
from PIL import Image

from SimpleHigherHRNet import SimpleHigherHRNet
from src.video.preprocessing.pose_extraction_utils import get_bboxes_for_frame, get_bbox_closest_to_previous_bbox, \
    apply_bbox_to_frame


def extract_pose_one_video(path_to_video: str, detector: object, output_path: str,
                           keep_the_same_person: Optional[bool] = False,
                           every_n_frame: Optional[int] = 1) -> pd.DataFrame:
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
    previous_pose = None
    last_bbox = None
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if (counter - 1) % every_n_frame == 0:
                # convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # calculate timestamp
                timestamp = counter * FPS_in_seconds
                # round it to 2 digits to make it readable
                timestamp = round(timestamp, 2)
                # recognize the pose
                bboxes =  get_bboxes_for_frame(extractor=detector, frame=frame)
                # create new filename for saving the face
                new_filename = video_filename
                new_filename = os.path.join(new_filename,
                                            f"{counter:05}.jpg")
                # if not recognized, note it as NaN
                if bboxes is None:
                    # if there were no face vefore
                    if previous_pose is None:
                        pose = np.zeros((224, 224, 3), dtype=np.uint8)
                        bbox = None
                    else:
                        # save previous face
                        pose = previous_pose
                        bbox = last_bbox
                else:
                    # otherwise, extract the face and save it
                    if keep_the_same_person and last_bbox is not None:
                        # if we want to keep the same person, then we need to calculate the distances between the
                        # center of the previous bbox and the current ones. Then, choose the closest one.
                        bbox = get_bbox_closest_to_previous_bbox(bboxes, last_bbox)
                    else:
                        # otherwise, take the first one, since we do not have confidences
                        bbox =  bboxes[0]
                    # extract face according to bbox
                    pose =  apply_bbox_to_frame(frame, bbox)
                # create full path to the output file
                output_filename = os.path.join(output_path, new_filename)
                # save extracted face
                Image.fromarray(pose).save(output_filename)
                metadata = pd.concat([metadata,
                                      pd.DataFrame.from_records([{
                                          "filename": output_filename,
                                          "frame_num": counter,
                                          "timestamp": timestamp}
                                      ])
                                      ], ignore_index=True)
                # save previous face and bbox
                previous_pose = pose
                last_bbox = bbox
            # increment counter
            counter += 1
        else:
            break
    return metadata


def extract_poses_from_all_videos(paths_to_videos: List[str], detector: object, output_path: str,
                                  keep_the_same_person: Optional[bool] = False,
                                  every_n_frame: Optional[int] = 1) -> pd.DataFrame:
    # create output directory if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    # create metadata file
    metadata = pd.DataFrame(columns=["filename", "frame_num", "timestamp"])
    # go through all videos
    for path_to_video in paths_to_videos:
        # extract faces from one video
        metadata_one_video =  extract_pose_one_video(path_to_video, detector, output_path,
                                                     keep_the_same_person, every_n_frame)
        # add metadata from one video to the main metadata
        metadata = pd.concat([metadata, metadata_one_video], ignore_index=True, axis=0)
        # print
        print(f"Finished processing {path_to_video}")
        # save metadata
        metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False)
    return metadata



if __name__ == "__main__":
    # TODO: check it
    path_to_data = r"F:\Datasets\AffWild2\videos"
    output_path = r"F:\Datasets\AffWild2\preprocessed\poses"
    # load detector
    detector = SimpleHigherHRNet(c=32, nof_joints=17,
                                 checkpoint_path="FILL_IN",
                                 return_heatmaps=False, return_bounding_boxes=True, max_batch_size=1, device="cuda")
    paths_to_videos = glob.glob(os.path.join(path_to_data, "*"))
    metadata = extract_poses_from_all_videos(paths_to_videos=paths_to_videos, detector=detector,
                                                output_path=output_path,
                                                keep_the_same_person=True, every_n_frame=1)

    # save metadata
    metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False)



