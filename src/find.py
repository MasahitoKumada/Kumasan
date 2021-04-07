import cv2
import dlib
import numpy as np
import os
import glob
import random

# random.seed(0)

# Cascade files directory path
CASCADE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/"
LEARNED_MODEL_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/learned-models/"
predictor = dlib.shape_predictor(
    LEARNED_MODEL_PATH + 'helen-dataset.dat')
face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH + 'haarcascade_frontalface_default.xml')

# In & Out: setting
INPUT_DIR = './input'
OUTPUT_DIR = './output'

# 顔の部位ラベル
INDEX_NOSE = 1
INDEX_RIGHT_EYEBROWS = 2
INDEX_LEFT_EYEBROWS = 3
INDEX_RIGHT_EYE = 4
INDEX_LEFT_EYE = 5
INDEX_OUTSIDE_LIPS = 6
INDEX_INSIDE_LIPS = 7
INDEX_CHIN = 8


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
                np.array(
                    [[p.x, p.y] for p in predictor(gray_img, rect).parts()])
            )
    return landmarks


def normalization(face_landmarks):
    return_list = []
    for facemark in face_landmarks:
        # nose
        nose = list(range(130, 147 + 1))
        nose.remove(139)

        # right eyebrows
        right_eyebrow = list(range(62, 83 + 1))
        right_eyebrow.remove(68)
        right_eyebrow.remove(79)

        # left eyebrows
        left_eyebrow = list(range(84, 105 + 1))
        left_eyebrow.remove(90)
        left_eyebrow.remove(101)

        # right eye
        right_eye = list(range(18, 39 + 1))
        right_eye.remove(24)
        right_eye.remove(35)

        # left_eye
        left_eye = list(range(40, 61 + 1))
        left_eye.remove(46)
        left_eye.remove(57)

        # outside lips
        outside_lips = list(range(148, 178 + 1))
        outside_lips.remove(150)
        outside_lips.remove(161)
        outside_lips.remove(172)

        # inside lips
        inside_lips = list(range(3, 17 + 1))
        inside_lips.remove(13)
        add_list_lips = list(range(179, 193 + 1))
        add_list_lips.remove(183)
        inside_lips += add_list_lips

        # chin
        chin = [0, 1, 106, 117, 128, 139, 150, 161, 172,
                183, 2, 13, 24, 35, 46, 57, 68, 79, 90, 101]
        add_list = list(range(107, 129 + 1))
        add_list.remove(117)
        add_list.remove(128)
        chin += add_list

        # nose
        for nose_i, fm_i in enumerate(nose):
            # nose[nose_i] = facemark[fm_i]
            nose[nose_i] = np.hstack([facemark[fm_i], INDEX_NOSE]) 
        # right_eyebrow
        for reb_i, fm_i in enumerate(right_eyebrow):
            # right_eyebrow[reb_i] = facemark[fm_i]
            right_eyebrow[reb_i] = np.hstack([facemark[fm_i], INDEX_RIGHT_EYEBROWS])
        # left_eyebrow
        for leb_i, fm_i in enumerate(left_eyebrow):
            # left_eyebrow[leb_i] = facemark[fm_i]
            left_eyebrow[leb_i] = np.hstack([facemark[fm_i], INDEX_LEFT_EYEBROWS])
        # right_eye
        for re_i, fm_i in enumerate(right_eye):
            # right_eye[re_i] = facemark[fm_i]
            right_eye[re_i] = np.hstack([facemark[fm_i], INDEX_RIGHT_EYE])
        # left_eye
        for le_i, fm_i in enumerate(left_eye):
            # left_eye[le_i] = facemark[fm_i]
            left_eye[le_i] = np.hstack([facemark[fm_i], INDEX_LEFT_EYE])
        # outside_lips
        for ol_i, fm_i in enumerate(outside_lips):
            # outside_lips[ol_i] = facemark[fm_i]
            outside_lips[ol_i] = np.hstack([facemark[fm_i], INDEX_OUTSIDE_LIPS])
        # inside_lips
        for il_i, fm_i in enumerate(inside_lips):
            # inside_lips[il_i] = facemark[fm_i]
            inside_lips[il_i] = np.hstack([facemark[fm_i], INDEX_INSIDE_LIPS])
        # chin
        for chin_i, fm_i in enumerate(chin):
            # chin[chin_i] = facemark[fm_i]
            chin[chin_i] = np.hstack([facemark[fm_i], INDEX_CHIN])

        return_list.append(chin + nose + outside_lips + inside_lips +
                           right_eye + left_eye + right_eyebrow + left_eyebrow)

    return return_list


def drow(img, points):

    if  points[2]==INDEX_NOSE:
        cv2.drawMarker(img, (points[0], points[1]), (21, 255, 12)) # (B, G, R): 緑
    elif points[2]==INDEX_RIGHT_EYEBROWS:
        cv2.drawMarker(img, (points[0], points[1]), (255, 0, 204)) # (B, G, R): ピンク
    elif points[2]==INDEX_LEFT_EYEBROWS:
        cv2.drawMarker(img, (points[0], points[1]), (255, 0, 204)) # (B, G, R): ピンク
    elif points[2]==INDEX_RIGHT_EYE:
        cv2.drawMarker(img, (points[0], points[1]), (0, 0, 204)) # (B, G, R): 赤
    elif points[2]==INDEX_LEFT_EYE:
        cv2.drawMarker(img, (points[0], points[1]), (0, 0, 204)) # (B, G, R): 赤
    elif points[2]==INDEX_OUTSIDE_LIPS:
        cv2.drawMarker(img, (points[0], points[1]), (255, 0, 0)) # (B, G, R): 青
    elif points[2]==INDEX_INSIDE_LIPS:
        cv2.drawMarker(img, (points[0], points[1]), (255, 0, 0)) # (B, G, R): 青
    elif points[2]==INDEX_CHIN:
        cv2.drawMarker(img, (points[0], points[1]), (0, 255, 255)) # (B, G, R): 緑


def make_face_part_candidate_lst(face_part_candidate_dic, index_part_name,  landmarks):   
    tmp_arr = np.array([0, 0]) # 初期化
    for landmark in landmarks:    
        for points in landmark:
            # drow(img, points)
            if points[2]==index_part_name:
                tmp_arr = np.vstack([
                    tmp_arr, [points[0], points[1]]
                    ])
    return tmp_arr[1:]


def main():
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR ,'*.jpg')))

    # print(image_paths)

    '''
    [0~40]: chin
    [41~57]: nose
    [58~85]: outside of lips
    [86-113]: inside of lips
    [114-133]: right eye
    [134-153]: left eye
    [154-173]: right eyebrows
    [174-193]: left eyebrows
    '''

    chin_lst = []
    nose_lst = []
    outside_lip_lst = []
    inside_lips_lst = []
    right_eye_lst = []
    left_eye_lst = []
    right_eyebrows_lst = []
    left_eyebrows_lst = []

    face_part_candidate_dic={
        'chin':chin_lst, 'nose':nose_lst, 'outside_lips':outside_lip_lst, 'inside_lips':inside_lips_lst,
        'right_eye':right_eye_lst, 'left_eye':left_eye_lst, 'right_eyebrows':right_eyebrows_lst,
        'left_eyebrows':left_eyebrows_lst
    }

    face_part_select_dic={
        'chin':chin_lst, 'nose':nose_lst, 'outside_lips':outside_lip_lst, 'inside_lips':inside_lips_lst,
        'right_eye':right_eye_lst, 'left_eye':left_eye_lst, 'right_eyebrows':right_eyebrows_lst,
        'left_eyebrows':left_eyebrows_lst
    }

    for image_path in image_paths:

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = facemark(gray)
        # print(landmarks)
        landmarks = normalization(landmarks)
        # print(landmarks)

        each_img_chin = make_face_part_candidate_lst(face_part_candidate_dic['chin'], INDEX_CHIN, landmarks)
        face_part_candidate_dic['chin'].append(each_img_chin)
        each_img_nose = make_face_part_candidate_lst(face_part_candidate_dic['nose'], INDEX_NOSE, landmarks)
        face_part_candidate_dic['nose'].append(each_img_nose)
        each_img_right_eyebroes = make_face_part_candidate_lst(face_part_candidate_dic['right_eyebrows'], INDEX_RIGHT_EYEBROWS, landmarks)
        face_part_candidate_dic['right_eyebrows'].append(each_img_right_eyebroes)
        each_img_left_eyebroes = make_face_part_candidate_lst(face_part_candidate_dic['left_eyebrows'], INDEX_LEFT_EYEBROWS, landmarks)
        face_part_candidate_dic['left_eyebrows'].append(each_img_left_eyebroes)  
        each_img_right_eye = make_face_part_candidate_lst(face_part_candidate_dic['right_eye'], INDEX_RIGHT_EYE, landmarks)
        face_part_candidate_dic['right_eye'].append(each_img_right_eye)
        each_img_left_eye = make_face_part_candidate_lst(face_part_candidate_dic['left_eye'], INDEX_LEFT_EYE, landmarks)
        face_part_candidate_dic['left_eye'].append(each_img_left_eye) 

        each_img_outside_lips = make_face_part_candidate_lst(face_part_candidate_dic['outside_lips'], INDEX_OUTSIDE_LIPS , landmarks)
        face_part_candidate_dic['outside_lips'].append(each_img_outside_lips) 
        each_img_inside_lips = make_face_part_candidate_lst(face_part_candidate_dic['inside_lips'], INDEX_INSIDE_LIPS , landmarks)
        face_part_candidate_dic['inside_lips'].append(each_img_inside_lips)        

        root, ext = os.path.splitext(image_path)
        cv2.imwrite(os.path.join(OUTPUT_DIR, root.split('/')[-1] + '_out' + ext), img)


    # print(face_part_candidate_dic)
    for k, v in face_part_candidate_dic.items():
        random_num = random.randint(0, len(v)-1)
        if len(v[random_num])==1:
            # 選択したパーツが空の場合は、もう一度リトライ
            random_num = random.randint(0, len(v)-1)
            face_part_select_dic[k]=v[random_num]
            
        else:
            face_part_select_dic[k]=v[random_num]
        

    #顔候補のリスト
    print(face_part_select_dic)
    

if __name__ == '__main__':
    main()
