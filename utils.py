import cv2
import os
import pandas as pd


def resize_dim(dim, fraction=1):
    w, h = dim[0], dim[1]
    if fraction == 1 and h > 500:
        fraction = 500 / h

    width = int(fraction * w)
    height = int(fraction * h)

    return (width, height)


def resize(img, dim):
    return cv2.resize(img, dim)


# check if key q was pressed
def check(c: str = 'q') -> bool:
    if cv2.waitKey(1) & 0xFF == ord(c):
        return True
    return False


def video_capture(value):
    return cv2.VideoCapture(value)


def image_position(name='Frame', x=20, y=20):
    cv2.namedWindow(name)
    cv2.moveWindow(name, x, y)


def show_img(title, img):
    cv2.imshow(title, img)


def destroy_windows():
    cv2.destroyAllWindows()


def draw_text(img, text):
    if not isinstance(text, str):
        text = str(text)
    blue = (255, 0, 0)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(img, text, (20, 70), font, 2, blue, 3)


def file_exists(path):
    return os.path.isfile(path)


def cvs(file, label, distances):
    columns = ['label', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    data = [label]
    for dist in distances[0]:
        data.append(dist[1])

    if file_exists(file):
        df = pd.read_csv(file)
        temp_df = pd.DataFrame([data], columns=columns, dtype=float)
        df = df.append(temp_df, ignore_index=False)
    else:
        df = pd.DataFrame(data=[data], columns=columns, dtype=float)

    df.to_csv(file, index=False)
