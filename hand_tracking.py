import mediapipe as mp
import cv2


class HandTracking:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=.5) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode, max_num_hands,
                                         min_detection_confidence, min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def bgr_to_rgb(self, img):
        # convert from bgr to rgb
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def landmark_positions(self, img):
        # make sure that img is RGB and not BGR
        img_rgb = self.bgr_to_rgb(img)
        # object with hand landmark pts
        results = self.hands.process(img_rgb)
        return results.multi_hand_landmarks

    def unit_to_img_pixels(self, dim, unit):
        # convert from unitari pts to image pixels
        pos_x, pos_y = int(unit[0] * dim[0]), int(unit[1] * dim[1])
        return (pos_x, pos_y)

    def draw_landmarks(self, img, hands_landamaks, lines=True):
        # draw lines connectin landmakrs
        for hand in hands_landamaks:
            if lines:
                self.mp_draw.draw_landmarks(
                    img, hand, self.mp_hands.HAND_CONNECTIONS)
            else:
                self.mp_draw.draw_landmarks(img, hand)

    # missing to get idx
    def get_landmark_idx(self, landmarks, idx):
        pass

    def get_landmark(self, dim, hand):
        '''
        returns a list containin the index of each landmark and is position [[idx, x, y],...]
        '''
        idx_pos = list()
        for idx, lm in enumerate(hand.landmark):
            # x, y = self.unit_to_img_pixels(dim, (lm.x, lm.y))
            idx_pos.append([idx, lm.x, lm.y])
        return idx_pos

    def get_all_landmarks(self, hands_landamaks, dim):
        landmarks = list()
        # for each hand
        if hands_landamaks:
            for hand in hands_landamaks:
                # getting idx and position for each landmark (0 - 1)
                landmark = self.get_landmark(dim, hand)
                landmarks.append(landmark)
        return landmarks

    def count_fingers(self, landmarks):
        finger_count = 0
        if len(landmarks) == 1:
            for hand in landmarks:
                # thumb
                if hand[4][2] < hand[5][2]:
                    finger_count += 1
                # index finger
                if hand[8][2] < hand[6][2]:
                    finger_count += 1
                # middle finger
                if hand[12][2] < hand[10][2]:
                    finger_count += 1
                # weding finger
                if hand[16][2] < hand[14][2]:
                    finger_count += 1
                # pinkey
                if hand[20][2] < hand[18][2]:
                    finger_count += 1
        elif len(landmarks) == 2:
            for hand in landmarks:
                # thumb
                if hand[4][2] < hand[5][2]:
                    finger_count += 1
                # index finger
                if hand[8][2] < hand[6][2]:
                    finger_count += 1
                # middle finger
                if hand[12][2] < hand[10][2]:
                    finger_count += 1
                # weding finger
                if hand[16][2] < hand[14][2]:
                    finger_count += 1
                # pinkey
                if hand[20][2] < hand[18][2]:
                    finger_count += 1
        return finger_count
