import time
import utils

from hand_tracking import HandTracking


def main():
    cap = utils.video_capture(0)
    width, height = cap.get(3), cap.get(4)

    win_name = 'Frame'
    utils.image_position(win_name, 20, 20)

    ht = HandTracking(max_num_hands=1)

    q_flag = False

    file = 'data/landmarks.csv'
    labels = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17,
              's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35}
    keys = list(labels.keys())

    while not q_flag:
        ret, frame = cap.read()

        dim = utils.resize_dim((width, height))
        frame = utils.resize(frame, dim)

        hands_landamaks = ht.landmark_positions(frame)

        if hands_landamaks:
            landmarks = ht.get_all_landmarks(hands_landamaks, dim)
            ht.draw_landmarks(frame, hands_landamaks)
            count = ht.count_fingers(landmarks)
            normalized = ht.normalize(landmarks)
            for key in keys:
                if utils.check(key):
                    utils.cvs(file, labels[key], normalized)
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    print(current_time, '- Saved Distances for:',
                          key, ':', labels[key])

        utils.show_img(win_name, frame)
        q_flag = utils.check('-')

    utils.destroy_windows()


if __name__ == '__main__':
    main()
