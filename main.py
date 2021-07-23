import time
import utils
from hand_tracking import HandTracking
import argparse
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_alpha", type=str,
                        help="choose [count, alpha]", default='alpha')
    parser.add_argument("--model_path", type=str,
                        help="Path to model", default='data/model.pth')
    args = parser.parse_args()

    if args.count_alpha == 'alpha':
        labels = utils.read_txt('data/labels.txt')
        print(labels)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(args.model_path)
        model.eval()
        model.to(device)

    cap = utils.video_capture(0)
    width, height = cap.get(3), cap.get(4)

    win_name = 'Frame'
    utils.image_position(win_name, 20, 20)

    ht = HandTracking()

    q_flag = False
    string = ''
    character = ''
    counter = 0
    while not q_flag:
        ret, frame = cap.read()

        dim = utils.resize_dim((width, height))
        frame = utils.resize(frame, dim)

        hands_landamaks = ht.landmark_positions(frame)
        if hands_landamaks:
            landmarks = ht.get_all_landmarks(hands_landamaks, dim)
            ht.draw_landmarks(frame, hands_landamaks)
            if args.count_alpha == 'count':
                count = ht.count_fingers(landmarks)
                utils.draw_text(frame, count)
            elif args.count_alpha == 'alpha':
                normalized = ht.normalize(landmarks)

                distance = np.asarray(normalized)
                distance = np.squeeze(distance)
                if len(distance.shape) == 2:
                    distance = distance[:, 1]
                    distance = distance.astype(float).reshape(1, -1)
                    distance = torch.from_numpy(distance)
                    distance.to(device)

                    y_hat = model(distance)
                    y_max = torch.argmax(y_hat, dim=1)
                    label = labels[int(y_max)]
                    if character == 'b':
                        string = ''
                        character = ''
                        counter = 0
                    if not counter:
                        character = labels[int(y_max)]
                    if label == character:
                        counter += 1
                        if counter == 20:
                            string += character
                    if character != label:
                        counter = 0

                    # utils.draw_text(frame, labels[int(y_max)])
                    utils.draw_text(frame, label, (20, 120), (0, 0, 255))
                    utils.draw_text(frame, string)

        utils.show_img(win_name, frame)
        q_flag = utils.check('q')

    utils.destroy_windows()


if __name__ == '__main__':
    main()
