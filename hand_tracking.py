import mediapipe as mp
import cv2


class HandTracking:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=.5) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode, max_num_hands,
                                         min_detection_confidence, min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def bgr_to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def landmark_positions(self, img):
        # make sure that img is RGB
        img_rgb = self.bgr_to_rgb(img)

        results = self.hands.process(img_rgb)

        return results.multi_hand_landmarks

    def unit_to_img_pixels(self, dim, unit):
        pos_x, pos_y = int(unit[0] * dim[0]), int(unit[1] * dim[1])
        return (pos_x, pos_y)

    def draw_landmarks(self, img, hand, lines=True):
        if lines:
            self.mp_draw.draw_landmarks(
                img, hand, self.mp_hands.HAND_CONNECTIONS)
        else:
            self.mp_draw.draw_landmarks(
                img, hand)

    # def get
    #  if results.multi_hand_landmarks:
    #       # for each hand
    #       for hand in results.multi_hand_landmarks:
    #            # geting idx and position for each landmark (0 - 1)
    #            for id, lm in enumerate(hand.landmark):
    #                 x, y = unit_to_img_pixels(dim, (lm.x, lm.y))

    #             mp_draw.draw_landmarks(
    #                 frame, hand, mp_hands.HAND_CONNECTIONS)
