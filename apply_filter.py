import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv

VISUALIZE_FACE_POINTS = False

filters_config = {
    'anonymous':
        [{'path': "filters/anonymous.png",
          'anno_path': "filters/anonymous.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anime':
        [{'path': "filters/anime.png",
          'anno_path': "filters/anime.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "filters/dog-ears.png",
          'anno_path': "filters/dog-ears.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/dog-nose.png",
          'anno_path': "filters/dog-nose.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "filters/cat-ears.png",
          'anno_path': "filters/cat-ears.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/cat-nose.png",
          'anno_path': "filters/cat-nose.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "filters/jason-joker.png",
          'anno_path': "filters/jason-joker.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'gold-crown':
        [{'path': "filters/gold-crown.png",
          'anno_path': "filters/gold-crown.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'flower-crown':
        [{'path': "filters/flower-crown.png",
          'anno_path': "filters/flower-crown.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}


# detect facial landmarks in image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime




if __name__=="__main":
    pass