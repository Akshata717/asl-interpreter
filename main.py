
import cv2
import mediapipe as mp
import csv
import os
import time
from collections import Counter
 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
 
SAVE_INTERVAL = 0.1   # seconds between samples while recording
 
# Load existing counts so you can see which letters need more data
counts = Counter()
if os.path.exists("dataset.csv"):
    with open("dataset.csv", newline="") as f:
        for r in csv.reader(f):
            if r:
                counts[r[-1]] += 1
 
cap = cv2.VideoCapture(0)
current = None
last_save = 0
 
with open("dataset.csv", mode="a", newline="") as f:
    writer = csv.writer(f)
 
    while True:
        success, frame = cap.read()
        if not success:
            break
 
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
 
        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # Esc quits
            break
        elif key == ord(' '):
            current = None                  # pause
        elif ord('a') <= key <= ord('z'):
            current = chr(key).upper()      # start recording this letter
 
        now = time.time()
        if current and hand_landmarks and now - last_save > SAVE_INTERVAL:
            row = [c for lm in hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)]
            row.append(current)
            writer.writerow(row)
            f.flush()
            counts[current] += 1
            last_save = now
 
        status = f"REC {current} ({counts[current]})" if current else "Paused - press a letter"
        color = (0, 0, 255) if current else (200, 200, 200)
        cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, "Letter=record  Space=pause  Esc=quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Collect", frame)
 
cap.release()
cv2.destroyAllWindows()
