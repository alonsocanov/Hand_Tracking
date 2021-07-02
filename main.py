import time
import utils
from hand_tracking import HandTracking


def main():
    cap = utils.video_capture(0)
    width, height = cap.get(3), cap.get(4)

    win_name = 'Frame'
    utils.image_position(win_name, 20, 20)

    ht = HandTracking()

    q_flag = False

    while not q_flag:
        ret, frame = cap.read()

        dim = utils.resize_dim((width, height))
        frame = utils.resize(frame, dim)

        hands_landamaks = ht.landmark_positions(frame)

        if hands_landamaks:
            landmarks = ht.get_all_landmarks(hands_landamaks, dim)
            ht.draw_landmarks(frame, hands_landamaks)
            count = ht.count_fingers(landmarks)
            utils.draw_text(frame, count)

        utils.show_img(win_name, frame)
        q_flag = utils.check('q')

    utils.destroy_windows()


if __name__ == '__main__':
    main()
