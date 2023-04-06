import glob
import os

import cv2
import mediapipe as mp
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

path_to_images = os.path.join("C:/Users/flamingo/Downloads/batch2/cropped_aligned_new_50_vids")

path_to_landmarks = os.path.join("C:/Users/flamingo/Downloads/batch2")


def calculate_triangle_area(param, param1, param2):
    """
    Calculates the area of a triangle using the three dimensional coordinates of the landmarks
    :param param: landmark
    :param param1: landmark
    :param param2: landmark
    :return: area of the triangle
    """
    a = (param.x - param1.x) * (param.y + param1.y)
    b = (param1.x - param2.x) * (param1.y + param2.y)
    c = (param2.x - param.x) * (param2.y + param.y)
    return 0.5 * abs(a + b + c)


def calculate_surface_area(landmarks):
    """
    Calculates the surface area of mouth using the three dimensional coordinates of the landmarks
    :param landmarks: list of landmarks
    :return: surface area of the face
    """
    landmarks = landmarks.landmark

    OUTER_LIPS = [78,191,80,81,82,13,312,311,310,415,308,78]

    # Landmark indices for the inner lips.
    INNER_LIPS = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78
    ]
    # Calculate the surface area using the three dimensional coordinates of the landmarks.
    surface_area = 0
    for i in range(len(OUTER_LIPS) - 1):
        surface_area += calculate_triangle_area(
            landmarks[OUTER_LIPS[i]], landmarks[INNER_LIPS[i]],
            landmarks[OUTER_LIPS[i + 1]])
        surface_area += calculate_triangle_area(
            landmarks[INNER_LIPS[i + 1]], landmarks[INNER_LIPS[i]],
            landmarks[OUTER_LIPS[i + 1]])
    return surface_area

def extract_surface_area():
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for folder in os.listdir(path_to_images):
            pd_lips = []
            for idx, file in enumerate(glob.glob(os.path.join(path_to_images, folder, '*.jpg'))):
                image = cv2.imread(file)
                file_name = os.path.basename(file).split(".")[0]
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                face_landmarks = results.multi_face_landmarks[0]

                surface_area = calculate_surface_area(face_landmarks)
                pd_lips.append([file_name,surface_area])
            pd_lips = pd.DataFrame(pd_lips, columns=['frame', 'surface_area_mouth'])

            mask = pd_lips[pd_lips["surface_area_mouth"].rolling(window=30).mean() > pd_lips["surface_area_mouth"].mean()]
            pd_lips["mouth_open"] = 0
            pd_lips.loc[mask.index, "mouth_open"] = 1
            save_path = os.path.join(path_to_landmarks, "features")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pd_lips.to_csv(os.path.join(save_path, folder + '.csv'), index=True)
            print('Done with folder: {}'.format(folder))


if    __name__ == '__main__':
    extract_surface_area()


