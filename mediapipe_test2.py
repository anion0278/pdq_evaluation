import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import time

# For static images:
IMAGE_FILES = [r"C:\Users\Stefan\Desktop\man-front-and-back-hand.webp"]
with mp_hands.Hands(
    static_image_mode=True,
    model_complexity = 0,
    max_num_hands=10,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 67))
    # Convert the BGR image to RGB before processing.
    
    st = time.time()
    test_count = 500
    for i in range(0, test_count):
        results = hands.process(image)
    et = time.time() - st
    print(et)


    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
    #   print('hand_landmarks:', hand_landmarks)
    #   print(
    #       f'Index finger tip coordinates: (',
    #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
    #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
    #   )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow(
        'annt', cv2.flip(annotated_image, 1))
    cv2.waitKey(0)
    # # Draw hand world landmarks.
    # if not results.multi_hand_world_landmarks:
    #   continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
    #   mp_drawing.plot_landmarks(
    #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

import cv2
import mediapipe as mp
# why do I need to do this, instead of simply 
# "mpHands = mp.solutions.hands"
# to access the hands function because otherwise it doesn't recognize the module
mpHands = mp.solutions.mediapipe.python.solutions.hands
Hands = mpHands.Hands() # doesn't recognize unless I do the entire line above

capture = cv2.VideoCapture(0)
while True:
    ret, img = capture.read()
    cv2.imshow("res", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
