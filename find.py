import cv2
import dlib
import numpy
import os

# assert os.path.isfile('IMG_1111_riku2.jpg')

# Cascade files directory path
CASCADE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/"
LEARNED_MODEL_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/"
    #helen-dataset.datのPATH
predictor = dlib.shape_predictor(
    LEARNED_MODEL_PATH + 'helen-dataset.dat')
        # openCVの中のdataフォルダのhaarcascade_frontalface_default.xmlのPATH
face_cascade = cv2.CascadeClassifier(
    '/home/kuribo/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def face_position(gray_img):
    """Detect faces position
    Return:
        faces: faces position list (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(gray_img, minSize=(100, 100))
    return faces


def facemark(gray_img):
    """Recoginize face landmark position by i-bug 300-w dataset
    Return:
        randmarks = [
        [x, y],
        [x, y],
        ...
        ]
        [0~40]: chin
        [41~57]: nose
        [58~85]: outside of lips
        [86-113]: inside of lips
        [114-133]: right eye
        [134-153]: left eye
        [154-173]: right eyebrows
        [174-193]: left eyebrows
    """
    faces_roi = face_position(gray_img)
    landmarks = []

    for face in faces_roi:
        x, y, w, h = face
        face_img = gray_img[y: y + h, x: x + w];

        detector = dlib.get_frontal_face_detector()
        rects = detector(gray_img, 1)

        landmarks = []
        for rect in rects:
            landmarks.append(
                numpy.array(
                    [[p.x, p.y] for p in predictor(gray_img, rect).parts()])
            )
    return landmarks


if __name__ == '__main__':
    cap = cv2.imread('IMG_001.jpg')
    # while cap.isOpened():
    # ret, frame = cap.read()
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    landmarks = facemark(gray)

    for landmark in landmarks:
        for points in landmark:
            cv2.drawMarker(cap, (points[0], points[1]), (21, 255, 12))
    cv2.imwrite("IMG.jpg", cap)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    # img = cv2.imread()

    # cap.release()
    # cv2.destroyAllWindows()
