import mediapipe as mp
import cv2


def init_hand_tracking(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=.5):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode, max_num_hands,
                           min_detection_confidence, min_tracking_confidence)
    mp_draw = mp.solutions.drawing_utils

    return mp_hands, hands, mp_draw


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def unit_to_img_pixels(dim, unit):
    pos_x, pos_y = int(unit[0] * dim[0]), int(unit[1] * dim[1])
    return (pos_x, pos_y)


def landmark_positions(img, hands):
    # make sure that img is RGB
    img_rgb = bgr_to_rgb(img)

    results = hands.process(img_rgb)

    return results.multi_hand_landmarks


# def get
#  if results.multi_hand_landmarks:
#       # for each hand
#       for hand in results.multi_hand_landmarks:
#            # geting idx and position for each landmark (0 - 1)
#            for id, lm in enumerate(hand.landmark):
#                 x, y = unit_to_img_pixels(dim, (lm.x, lm.y))

#             mp_draw.draw_landmarks(
#                 frame, hand, mp_hands.HAND_CONNECTIONS)
