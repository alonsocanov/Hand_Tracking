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
    labels = {'a': 0, 'b': 1, 'c': 2}
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
            # if utils.check('d'):
            #     utils.delete_last_entry(file)
            # utils.draw_text(frame, count)

        utils.show_img(win_name, frame)
        q_flag = utils.check('q')

    utils.destroy_windows()


if __name__ == '__main__':
    main()
