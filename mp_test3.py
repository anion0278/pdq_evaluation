import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2 as mediapipe_landmarks     

mediapipe_drawing = mp.solutions.drawing_utils               
mediapipe_drawing_styles = mp.solutions.drawing_styles       
mediapipe_pose = mp.solutions.pose                           
bodyconnections = [(0,1), (0,6), (0,2), (2,4), (1,7), (1,3), (3,5), 
(6,7), (6,8), (8,10), (7,9), (9,11)]


IMAGE_FILES = [r"C:\Users\Stefan\Desktop\3.jpg"]
BG_COLOR = (192, 192, 192) 
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated_image = image.copy()

        landmark_subset = mediapipe_landmarks.NormalizedLandmarkList(
        landmark=[   
            results.pose_landmarks.landmark[11],
            results.pose_landmarks.landmark[12],
            results.pose_landmarks.landmark[13],
            results.pose_landmarks.landmark[14],
            results.pose_landmarks.landmark[15],
            results.pose_landmarks.landmark[16],
            results.pose_landmarks.landmark[23],
            results.pose_landmarks.landmark[24],
            results.pose_landmarks.landmark[25],
            results.pose_landmarks.landmark[26],
            results.pose_landmarks.landmark[27],
            results.pose_landmarks.landmark[28]              ]  )
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mediapipe_drawing.draw_landmarks(
            image,
            landmark_subset
            )
        
        poses = landmark_subset.landmark
        for i in range(0, len(bodyconnections)):
            start_idx = [
                poses[bodyconnections[i][0]].x,
                poses[bodyconnections[i][0]].y
            ]

            end_idx = [
                poses[bodyconnections[i][1]].x,
                poses[bodyconnections[i][1]].y
            ]
            IMG_HEIGHT, IMG_WIDTH = image.shape[:2]

            cv2.line(image,
                tuple(np.multiply(start_idx[:2], [
                    IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                tuple(np.multiply(end_idx[:2], [
                    IMG_WIDTH, IMG_HEIGHT]).astype(int)),
                (255, 0, 0), 9)
        
        cv2.imshow('Pose_Check', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
