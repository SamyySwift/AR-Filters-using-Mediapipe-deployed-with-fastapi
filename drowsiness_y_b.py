from multiprocessing import connection
import cv2 as  cv
import mediapipe as mp
from scipy.spatial import distance as dist

counter = 0
total_blinks = 0
ear_consec_frames = 3
total = 0
drowsy_detected = 0


mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

Right_Eye = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
Left_Eye = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
Lips = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

def getLandmarks(image, results, draw=False):
    height, width = image.shape[:2]
    # for face_landmarks in results.multi_face_landmarks:
    #             mp_draw.draw_landmarks(
    #                 image = image,
    #                 landmark_list = face_landmarks,
    #                 landmark_drawing_spec = None,
    #                 connections = mp_face_mesh.FACEMESH_TESSELATION,
    #                 connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
    mesh_coords = [(int(point.x * width), int(point.y*height)) for point in results.multi_face_landmarks[0].landmark]

    return mesh_coords

def get_ear(img, landmarks, right_indices, left_indices):
    # Get the corordinates of the horinzontal left eye
    leh_right = landmarks[left_indices[0]]
    leh_left = landmarks[left_indices[8]]

    # Get the corordinates of the vertical left eye
    lev_top = landmarks[left_indices[12]]
    lev_bottom = landmarks[left_indices[4]]

    # Get the corordinates of the horinzontal right eye
    reh_right = landmarks[right_indices[0]]
    reh_left = landmarks[right_indices[8]]

    # Get the corordinates of the vertical right eye
    rev_top = landmarks[right_indices[12]]
    rev_bottom = landmarks[right_indices[4]]

    # cv.line(img, reh_right, reh_left, (255, 0, 0), 1)
    # cv.line(img, rev_top, rev_bottom, (0, 255, 0), 1)

    # cv.line(img, leh_right, leh_left, (255, 0, 0), 1)
    # cv.line(img, lev_top, lev_bottom, (0, 255, 0), 1)

    # Calculate the distance between the two vertical and horizontal lines for the left eye
    leh_dist = dist.euclidean(leh_right, leh_left)
    lev_dist = dist.euclidean(lev_top, lev_bottom)
    

    # Calculate the distance between the two vertical and horizontal lines for the right eye
    reh_dist = dist.euclidean(reh_right, reh_left)
    rev_dist = dist.euclidean(rev_top, rev_bottom)

    # Calculate the ratio of the distance between the two vertical lines
    re_ratio = rev_dist / reh_dist
    le_ratio = lev_dist / leh_dist

    ratio = (re_ratio + le_ratio) / 2

    return ratio

def get_mar(landmarks, lip_indices):

    #Get the co-ordinates of the horizontal distance of the lips 
    horizontal_l = landmarks[lip_indices[0]]
    horizontal_r = landmarks[lip_indices[10]]

    #Get the co-ordinates of the left vertical distance of the lips
    ver_l_top = landmarks[lip_indices[23]]
    ver_l_bot = landmarks[lip_indices[3]]

    #Get the co-ordinates of the middle vertical distance of the lips
    ver_m_top = landmarks[lip_indices[25]]
    ver_m_bot = landmarks[lip_indices[5]]

    #Get the co-ordinates of the right vertical distance of the lips
    ver_r_top = landmarks[lip_indices[27]]
    ver_r_bot = landmarks[lip_indices[7]]

    #Calculate the horizontal distance 

    hor_dist = dist.euclidean(horizontal_l,horizontal_r)

    #Calculate the vertical distances

    ver_l_dist = dist.euclidean(ver_l_top,ver_l_bot)
    ver_m_dist = dist.euclidean(ver_m_top,ver_m_bot)
    ver_r_dist = dist.euclidean(ver_r_top,ver_r_bot) 

    #calculate mouth aspect ratio (mar)
    total_ver = (ver_l_dist + ver_m_dist + ver_r_dist) / 3 
    mar = total_ver/hor_dist 

    return mar 





drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)
# For Webcam input
cap = cv.VideoCapture(0)
fps =  cap.get(cv.CAP_PROP_FPS)
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while True:
        success, image = cap.read()
        if not success:
            print("Failed to read frame")
            continue

        # Improve performance
        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image) 
       

        # Draw the results
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            mesh_coords = getLandmarks(image, results, False)
            # [cv.circle(image, point, 1, (0, 255, 0), -1) for point in [mesh_coords[i] for i in Lips]]
            # [cv.circle(image, point, 1, (0, 255, 0), -1) for point in [mesh_coords[i] for i in Left_Eye]]
            # [cv.circle(image, point, 1, (0, 255, 0), -1) for point in [mesh_coords[i] for i in Right_Eye]]

            ratio = get_ear(image, mesh_coords, Right_Eye, Left_Eye)
            mar_ratio = get_mar(mesh_coords,Lips) 
            if mar_ratio > 0 :
                total += 1
            cv.putText(image, f'mar_ratio: {round(mar_ratio, 3),total}', (100, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
            cv.putText(image, f'ear_ratio: {round(ratio, 3)}', (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
            if (ratio < 0.3) | (mar_ratio > 0.7) :
                counter += 1
                
                if (counter > 3) & (total >= fps * 5 ):
                    cv.putText(image, 'You are drowsy', (150, 30), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 3)
                    drowsy_detected += 1
                    total = 0
                    counter = 0
                elif (total >= fps *5 ):
                    total = 0
                    counter = 0
            cv.putText(image, f'{drowsy_detected} drowsiness detections', (350, 30), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        cv.imshow("Face Mesh", image)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
