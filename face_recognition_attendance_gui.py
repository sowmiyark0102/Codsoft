import face_recognition
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk

# -----------------------------
# 📁 LOAD & ENCODE FACES
# -----------------------------
dataset_path = "dataset"

known_encodings = []
known_names = []

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)

    for img in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

print("✅ Face encoding completed")

# -----------------------------
# 📊 ATTENDANCE SETUP
# -----------------------------
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(attendance_file, index=False)

marked_today = set()

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    key = f"{name}_{today}"

    if key not in marked_today:
        marked_today.add(key)

        df = pd.DataFrame([[name, today, time]],
                          columns=["Name", "Date", "Time"])
        df.to_csv(attendance_file, mode="a", header=False, index=False)

# -----------------------------
# 🖥️ GUI SETUP
# -----------------------------
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("850x650")

video_label = tk.Label(root)
video_label.pack()

cap = cv2.VideoCapture(0)

running = False

# -----------------------------
# START CAMERA
# -----------------------------
def start_camera():
    global running
    running = True
    update_frame()

# -----------------------------
# STOP CAMERA
# -----------------------------
def stop_camera():
    global running
    running = False

# -----------------------------
# FRAME PROCESSING
# -----------------------------
process_frame = True

def update_frame():
    global process_frame

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb = small_frame[:, :, ::-1]

    if process_frame:
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        face_names = []

        for encoding in face_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)

            name = "Unknown"

            if len(distances) > 0:
                min_dist = np.min(distances)

                if min_dist < 0.5:
                    index = np.argmin(distances)
                    name = known_names[index]
                    mark_attendance(name)

            face_names.append(name)

    process_frame = not process_frame

    # DRAW RESULTS
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        color = (0, 255, 0) if name 

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert for GUI
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    video_label.imgtk = img
    video_label.configure(image=img)

    video_label.after(10, update_frame)

start_btn = tk.Button(root, text="Start Camera", bg="green", fg="white", command=start_camera)
start_btn.pack(pady=10)

stop_btn = tk.Button(root, text="Stop Camera", bg="red", fg="white", command=stop_camera)
stop_btn.pack(pady=10)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
