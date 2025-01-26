import pickle
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import tempfile
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

st.title("X-Pose")

st.header("Select Video File")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

process_button = st.button("Process Video")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_midpoint(coord1, coord2):
    x = (coord1[0] + coord2[0]) / 2
    y = (coord1[1] + coord2[1]) / 2
    z = (coord1[2] + coord2[2]) / 2
    return np.array([x, y, z])


def calculate_angle_using_cross_product(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cross_product = np.cross(ba, bc)
    dot_product = np.dot(ba, bc)
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
    return np.degrees(angle)


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def calculate_angle_2d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

puttext_counters = {
        "elbow_fi": 0,
        "elbow_hi": 0,
        "arm_st": 0,
        "elbow_wi": 0,
        "elbow_cl": 0,
        "wrist_be": 0,
        "wrist_cl": 0,
        "ankle_cl": 0,
        "ankle_wi": 0,
        "knee_fa": 0,
    }

error_flags = {
        "elbow_fi": 0,
        "elbow_hi": 0,
        "arm_st": 0,
        "elbow_wi": 0,
        "elbow_cl": 0,
        "wrist_be": 0,
        "wrist_cl": 0,
        "ankle_cl": 0,
        "ankle_wi": 0,
        "knee_fa": 0,
    }

def set_error(error_type, frame, active_errors=None):
    if active_errors is None:
        active_errors = [error_type]

    # Reset specific error flags and counters if they are not in the active error types
    for key in error_flags.keys():
        if key not in active_errors:
            error_flags[key] = 0
            puttext_counters[key] = 0

    # Set the current error flag and counter
    error_flags[error_type] = 1
    puttext_counters[error_type] = frame

model = load_model('lstm_model2.h5')

data = pd.read_csv('csv2/output (1).csv')
label_encoder = LabelEncoder()
test_label_encoder = joblib.load('label_encoder (1).pkl')

data['class_name'] = label_encoder.fit_transform(data['class_name'])

scaler = StandardScaler()
scaler.fit(data[['right_shoulder_angle',
            'right_elbow_angle',
            'right_hip_angle',
            'right_knee_angle',
            'left_shoulder_angle',
            'left_elbow_angle',
            'left_hip_angle',
            'left_knee_angle']])

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)


def process_video(uploaded_file):
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_file_path)

    if not cap.isOpened():
        st.error("Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    progress_bar = st.progress(0)

    frame_count = 0
    ex_elbo_dis = [0.0, 0.0, 0.0]
    lat_max = []
    squat_angle = 1.0
    squat_stand = 0.0
    elbow_angle_temp = 0.0
    elbow_x = 0.0
    puttext = 0

    error_el = 0
    error_el_hi = 0

    frame_data = []
    predictions = []
    sequence_length = 40
    timesteps = 40

    puttext_counters = {
        "elbow_fi": 0,
        "elbow_hi": 0,
        "arm_st": 0,
        "elbow_wi": 0,
        "elbow_cl": 0,
        "wrist_be": 0,
        "wrist_cl": 0,
        "ankle_cl": 0,
        "ankle_wi": 0,
        "knee_fa": 0,
    }
    error_flags = {
        "elbow_fi": 0,
        "elbow_hi": 0,
        "arm_st": 0,
        "elbow_wi": 0,
        "elbow_cl": 0,
        "wrist_be": 0,
        "wrist_cl": 0,
        "ankle_cl": 0,
        "ankle_wi": 0,
        "knee_fa": 0,
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_file:
        output_path = tmp_output_file.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)

            for error_type, puttext in puttext_counters.items():

                if error_type == "elbow_fi":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, f'FIX Elbow', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "elbow_hi":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, f'Elbow too high', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "arm_st":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, f'Keep the support arm straight', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "elbow_wi":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, f'Too wide elbow!', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "elbow_cl":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, f'Elbow too close to torso', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "wrist_be":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, f'Wrist too bent!', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "wrist_cl":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, "Keep Wrists Closer", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "ankle_cl":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, "Keep Ankles Closer", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "ankle_wi":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, "Keep Ankles Wider", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0
                elif error_type == "knee_fa":
                    if puttext != 0 and error_flags[error_type] != 0:
                        cv2.putText(frame, "Knee too far", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        puttext_counters[error_type] -= 1
                        if puttext_counters[error_type] == 0:
                            error_flags[error_type] = 0

            if puttext != 0 and error_el != 0:
                cv2.putText(frame, f'FIX Elbow', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                puttext -= 1
                if puttext == 0:
                    error_el = 0

            if pose_results.pose_landmarks:

                landmarks = pose_results.pose_landmarks.landmark

                try:
                    left_shoulder_x = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow_x = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist_x = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_hip_x = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee_x = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle_x = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    right_shoulder_x = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow_x = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist_x = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_knee_x = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle_x = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_hip_x = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    right_shoulder_angle = calculate_angle_2d(right_hip_x, right_shoulder_x, right_elbow_x)
                    right_elbow_angle = calculate_angle_2d(right_shoulder_x, right_elbow_x, right_wrist_x)
                    right_hip_angle = calculate_angle_2d(right_knee_x, right_hip_x, right_shoulder_x)
                    right_knee_angle = calculate_angle_2d(right_hip_x, right_knee_x, right_ankle_x)

                    left_shoulder_angle = calculate_angle_2d(left_hip_x, left_shoulder_x, left_elbow_x)
                    left_elbow_angle = calculate_angle_2d(left_shoulder_x, left_elbow_x, left_wrist_x)
                    left_hip_angle = calculate_angle_2d(left_knee_x, left_hip_x, left_shoulder_x)
                    left_knee_angle = calculate_angle_2d(left_hip_x, left_knee_x, left_ankle_x)

                except IndexError:
                    right_shoulder_angle = np.nan
                    right_elbow_angle = np.nan
                    right_hip_angle = np.nan
                    right_knee_angle = np.nan
                    left_shoulder_angle = np.nan
                    left_elbow_angle = np.nan
                    left_hip_angle = np.nan
                    left_knee_angle = np.nan
            else:
                right_shoulder_angle = np.nan
                right_elbow_angle = np.nan
                right_hip_angle = np.nan
                right_knee_angle = np.nan
                left_shoulder_angle = np.nan
                left_elbow_angle = np.nan
                left_hip_angle = np.nan
                left_knee_angle = np.nan
            frame_data.append([right_shoulder_angle, right_elbow_angle, right_hip_angle,
                               right_knee_angle, left_shoulder_angle, left_elbow_angle, left_hip_angle,
                               left_knee_angle, ])


            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if len(frame_data) >= sequence_length:
                sequence = scaler.transform(frame_data[-sequence_length:])
                sequence = np.expand_dims(sequence, axis=0)
                prediction = model.predict(sequence)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_label = label_encoder.inverse_transform(predicted_class)[0]
                predictions.append(predicted_label)

                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

                right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]

                names = label_encoder.classes_

                if predicted_label == names[0]:  # bicep
                    if left_shoulder.visibility > 0.5:
                        if left_shoulder.visibility > 0.5:
                            if frame_count > 5:
                                sho_hip_dis = calculate_distance(
                                    [left_hip.x, left_hip.y, left_hip.z],
                                    [left_shoulder.x, left_shoulder.y, left_shoulder.z])
                                if elbow_x > left_elbow.x:
                                    if abs(elbow_x - left_elbow.x) > sho_hip_dis * 0.1:
                                        set_error("elbow_fi", 15)
                                        cv2.putText(frame, f'FIX Elbow', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 0), 2)
                                else:
                                    if abs(left_elbow.x - elbow_x) > sho_hip_dis * 0.1:
                                        set_error("elbow_fi", 15)
                                        cv2.putText(frame, f'FIX Elbow', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 255, 0), 2)
                            if frame_count % 5 == 0:
                                elbow_x = left_elbow.x
                        if frame_count % 5 == 0:
                            elbow_x = left_elbow.x
                    else:

                        if frame_count > 5:

                            sho_hip_dis = calculate_distance(
                                [right_hip.x, right_hip.y, right_hip.z],
                                [right_shoulder.x, right_shoulder.y, right_shoulder.z])

                            if elbow_x > right_elbow.x:
                                if abs(elbow_x - right_elbow.x) > sho_hip_dis * 0.1:
                                    set_error("elbow_fi", 15)
                                    cv2.putText(frame, f'FIX Elbow', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
                            else:
                                if abs(right_elbow.x - elbow_x) > sho_hip_dis * 0.1:
                                    set_error("elbow_fi", 15)
                                    cv2.putText(frame, f'FIX Elbow', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)

                        if frame_count % 5 == 0:  # 2 sec
                            elbow_x = right_elbow.x

                elif predicted_label == names[3]:  # dumbbell row
                    predicted_label = "dumbbell row"

                    leftangle2 = calculate_angle_2d(
                        [left_shoulder.x, left_shoulder.y],
                        [left_elbow.x, left_elbow.y],
                        [left_wrist.x, left_wrist.y]
                    )
                    rightangle2 = calculate_angle_2d(
                        [right_shoulder.x, right_shoulder.y],
                        [right_elbow.x, right_elbow.y],
                        [right_wrist.x, right_wrist.y]
                    )

                    if leftangle2 < rightangle2:
                        back_mid = calculate_midpoint([left_shoulder.x, left_shoulder.y, left_shoulder.z],
                                                      [left_hip.x, left_hip.y, left_hip.z])

                        if left_elbow.y * 1.05 < back_mid[1]:
                            set_error("elbow_hi", 7, ["elbow_hi", "arm_st"])
                            cv2.putText(frame, f'Elbow too high', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)

                        if rightangle2 < np.degrees(2.26892803):  # 130 degree
                            set_error("arm_st", 7, ["elbow_hi", "arm_st"])
                            cv2.putText(frame, f'Keep the support arm straight', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)
                    else:
                        back_mid = calculate_midpoint([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                      [right_hip.x, right_hip.y, right_hip.z])

                        if right_elbow.y * 1.05 < back_mid[1]:
                            set_error("elbow_hi", 7, ["elbow_hi", "arm_st"])
                            cv2.putText(frame, f'Elbow too high', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)

                        if leftangle2 < np.degrees(2.26892803):
                            set_error("arm_st", 7, ["elbow_hi", "arm_st"])
                            cv2.putText(frame, f'Keep the support arm straight', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)

                elif predicted_label == names[1]:  # Incline Dumbbell Press
                    predicted_label = "Incline Dumbbell Press"
                    angle = calculate_angle(
                        [left_shoulder.x, left_shoulder.y, left_shoulder.z],
                        [left_elbow.x, left_elbow.y, left_elbow.z],
                        [left_wrist.x, left_wrist.y, left_wrist.z]
                    )
                    if left_elbow.y * 0.95 > left_shoulder.y:
                        shoulder_angle = calculate_angle([right_shoulder.x, right_shoulder.y, right_shoulder.z],
                                                         [right_elbow.x, right_elbow.y, right_elbow.z],
                                                         [right_hip.x, right_hip.y, right_hip.z])

                        # Pose correction: Check if shoulder angle is too wide (indicating improper elbow position)
                        if shoulder_angle > np.degrees(2.7925268):
                            set_error("elbow_wi", 7, ["elbow_wi", "elbow_cl", "wrist_be"])
                            cv2.putText(frame, f'Too wide elbow!', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        2)
                        # Pose correction: Check for a proper range of motion (ROM)
                        elif shoulder_angle < np.degrees(1.0471976):
                            set_error("elbow_cl", 7, ["elbow_wi", "elbow_cl", "wrist_be"])
                            cv2.putText(frame, f'Elbow too close to torso', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)

                    wrist_angle = calculate_angle(
                        [left_elbow.x, left_elbow.y, left_elbow.z],
                        [left_wrist.x, left_wrist.y, left_wrist.z],
                        [right_wrist.x, right_wrist.y, right_wrist.z]
                    )

                    if wrist_angle < 80:
                        set_error("wrist_be", 7, ["elbow_wi", "elbow_cl", "wrist_be"])
                        cv2.putText(frame, f'Wrist too bent!', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif predicted_label == names[2]:  # Lat Pull Down
                    predicted_label = "Lat Pull Down"

                    shoulder_distance = calculate_distance(
                        [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                        [left_shoulder.x, left_shoulder.y, left_shoulder.z]
                    )
                    wrist_distance = calculate_distance(
                        [right_wrist.x, right_wrist.y, right_wrist.z],
                        [left_wrist.x, left_wrist.y, left_wrist.z]
                    )
                    if wrist_distance > shoulder_distance * 2.5:
                        set_error("wrist_cl", 7)
                        cv2.putText(frame, "Keep Wrists Closer", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif predicted_label == names[4]:  # Squat

                    shoulder_distance = calculate_distance(
                        [right_shoulder.x, right_shoulder.y, right_shoulder.z],
                        [left_shoulder.x, left_shoulder.y, left_shoulder.z]
                    )
                    ankle_distance = calculate_distance(
                        [right_ankle.x, right_ankle.y, right_ankle.z],
                        [left_ankle.x, left_ankle.y, left_ankle.z]
                    )

                    if right_knee.visibility > 0.5:
                        squat_angle = calculate_angle(
                            [right_hip.x, right_hip.y, right_hip.z],
                            [right_knee.x, right_knee.y, right_knee.z],
                            [right_ankle.x, right_ankle.y, right_ankle.z]
                        )
                        squat_angle2 = calculate_angle_2d(
                            [right_hip.x, right_hip.y],
                            [right_knee.x, right_knee.y],
                            [right_ankle.x, right_ankle.y]
                        )
                        if squat_angle > np.degrees(2.7925268):  # 160 degree
                            if squat_angle2 > np.degrees(2.7925268):
                                if right_knee.x < right_foot_index.x:
                                    squat_stand = abs(right_foot_index.x - right_knee.x) * 1.3
                                else:
                                    squat_stand = abs(right_knee.x - right_foot_index.x) * 1.3

                            if ankle_distance > (shoulder_distance * 1.43):
                                set_error("ankle_cl", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                cv2.putText(frame, f'Adjust your ankle closer', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                            if ankle_distance < (shoulder_distance * 0.8):
                                set_error("ankle_wi", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                cv2.putText(frame, f'Adjust your ankle wider', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                        if squat_stand != 0:
                            if right_knee.x < right_foot_index.x:
                                if right_knee.x + squat_stand < right_foot_index.x:
                                    set_error("knee_fa", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                    cv2.putText(frame, f'False knee position', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
                            else:
                                if right_knee.x - squat_stand > right_foot_index.x:
                                    set_error("knee_fa", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                    cv2.putText(frame, f'False knee position', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
                    else:
                        squat_angle = calculate_angle(
                            [left_hip.x, left_hip.y, left_hip.z],
                            [left_knee.x, left_knee.y, left_knee.z],
                            [left_ankle.x, left_ankle.y, left_ankle.z]
                        )
                        if squat_angle > np.degrees(2.7925268):  # 160 degree
                            if squat_angle2 > np.degrees(2.7925268):
                                if left_knee.x < left_foot_index.x:
                                    squat_stand = abs(left_foot_index.x - left_knee.x) * 1.3
                                else:
                                    squat_stand = abs(left_knee.x - left_foot_index.x) * 1.3
                            if ankle_distance > (shoulder_distance * 1.43):
                                set_error("ankle_cl", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                cv2.putText(frame, f'Adjust your ankle closer', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                            if ankle_distance < (shoulder_distance * 0.8):
                                set_error("ankle_wi", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                cv2.putText(frame, f'Adjust your ankle wider', (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

                        if squat_stand != 0:
                            if left_knee.x < left_foot_index.x:
                                if left_knee.x + squat_stand < left_foot_index.x:
                                    set_error("knee_fa", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                    cv2.putText(frame, f'False knee position', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
                            else:
                                if left_knee.x - squat_stand > left_knee.x:
                                    set_error("knee_fa", 7, ["ankle_cl", "knee_fa", "ankle_wi"])
                                    cv2.putText(frame, f'False knee position', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)

                # Overlay prediction on the video frame
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            progress_bar.progress(int(frame_count / total_frames * 100))

            # Write the frame to the output video
            out.write(frame)

    finally:
        cap.release()
        out.release()

        import time
        time.sleep(1)
        # Display the processed video
        st.success("Video processing is complete!")
        # st.video(output_path)

        # Provide download option
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4",
        )

        os.remove(tmp_file_path)
        os.remove(output_path)


if uploaded_file is not None and process_button:
    process_video(uploaded_file)
elif uploaded_file is None:
    st.error("Please upload a video file.")
