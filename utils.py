import cv2


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


def show_img(title, img):
    cv2.imshow(title, img)


def destroy_windows():
    cv2.destroyAllWindows()
