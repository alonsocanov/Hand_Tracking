import time
import utils
import hand_tracking as ht


def main():
    cap = utils.video_capture(0)
    width, height = cap.get(3), cap.get(4)

    mp_hands, hands, mp_draw = ht.init_hand_tracking()

    q_flag = False

    while not q_flag:
        ret, frame = cap.read()

        dim = utils.resize_dim((width, height))
        frame = utils.resize(frame, dim)

        hands_landamaks = ht.landmark_positions(frame, hands)

        if hands_landamaks:
            # for each hand
            for hand in hands_landamaks:
                # geting idx and position for each landmark (0 - 1)
                for id, lm in enumerate(hand.landmark):
                    x, y = ht.unit_to_img_pixels(dim, (lm.x, lm.y))

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS)

        utils.show_img('frame', frame)
        q_flag = utils.check('q')

    utils.destroy_windows()


if __name__ == '__main__':
    main()
