import os
import cv2
import face_recognition
import glob 
import time
import numpy as np
import pickle
from scipy.spatial import distance as dist


# Define image path
img_path = 'images'


def load_encoding_images(images_path):

    if os.path.exists('database/peopledb.pickle'):
        with open('database/peopledb.pickle', 'rb') as file:
            user_db = pickle.load(file)
    else:
        user_db = {}

        # Store image encoding and names
        for img_path in glob.glob(images_path +'/*'):
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract name from file path
            name = os.path.splitext(os.path.basename(img_path))[0]
            
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            user_db[name] = img_encoding

        os.makedirs('database')
        with open('database/peopledb.pickle', 'wb') as file:
            pickle.dump(user_db, file, protocol=pickle.HIGHEST_PROTOCOL)


    return user_db


# load encodings and create database
peopledb = load_encoding_images(img_path)


def get_ear(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def recognize(img, ear, frame):
    
    face_locations = []
    face_encodings = []
    face_names = []
    
    if ear > 0.15:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(list(peopledb.values()), face_encoding)
            name = "Unknown Person"
            
             # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(list(peopledb.values()), face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = list(peopledb.keys())[best_match_index]

            face_names.append(name)
    else:
        cv2.putText(frame, 'No Liveliness Detected', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255, 255), 1)

      
   
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255, 255), 1)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, name, (left+6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    return face_names
