from typing import List, Union, Tuple
from PIL import Image
import numpy as np



def apply_bbox_to_frame(frame, bbox)->np.ndarray:
    # define adding factor and width and height
    adding_factor = 15
    height, width, _ = frame.shape
    tmp_bbox = bbox.copy()
    # expand bbox so that it will cover all human with some space
    # height
    tmp_bbox[1] -= adding_factor
    tmp_bbox[3] += adding_factor
    # width
    tmp_bbox[0] -= adding_factor
    tmp_bbox[2] += adding_factor
    # check if we are still in the frame
    if tmp_bbox[1] < 0:
        tmp_bbox[1] = 0
    if tmp_bbox[3] > height:
        tmp_bbox[3] = height
    if tmp_bbox[0] < 0:
        tmp_bbox[0] = 0
    if tmp_bbox[2] > width:
        tmp_bbox[2] = width

    # cut frame
    tmp_bbox = [int(x) for x in tmp_bbox]
    cut_frame = frame[tmp_bbox[1]:tmp_bbox[3], tmp_bbox[0]:tmp_bbox[2]]
    return cut_frame


def get_bboxes_for_frame(extractor:object, frame: np.ndarray) -> Union[List[List[int]], None]:
    prediction = extractor.predict(frame)
    if prediction is None or len(prediction[0]) == 0:
        return None
    bboxes = [item.squeeze() for item in prediction[0]] # we use mode with single person, therefore there is no need to take bboxes from different persons
    return bboxes


def get_bbox_closest_to_previous_bbox(bboxes:Tuple[List[float],...], previous_bbox:List[float])->List[float]:
    """ Finds the bounding box, which is closest to the previous bounding box.

    :param faces_bboxes: Tuple[List[float],...]
            List of bounding boxes of faces, which were recognized by RetinaFace model
    :param previous_bbox: List[float]
            List of 5 floats, which represent the bounding box of face and its confidence
    :return: List[float]
            Closest to the previous_bbox bbox
    """
    # find the center of the previous bbox. The order of the coordinates is (x1, y1, x2, y2) = bbox[:4]
    previous_bbox_center = ((previous_bbox[0]+previous_bbox[2])/2, (previous_bbox[1]+previous_bbox[3])/2)
    # find the centers of all bboxes. The order of the coordinates is (x1, y1, x2, y2) = bbox[:4]
    bboxes_centers = []
    for bbox in bboxes:
        bboxes_centers.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
    # find the bbox, which is closest to the previous bbox based on the euclidean distance
    distances = [np.linalg.norm(
                np.array(previous_bbox_center)-np.array((bbox_center_x, bbox_center_y))
                )
                for bbox_center_x, bbox_center_y in bboxes_centers]
    idx_closest_bbox = np.argmin(distances) # TODO: check it
    closest_bbox = bboxes[idx_closest_bbox]

    return closest_bbox




