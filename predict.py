
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque, Counter
 

with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)
 
 
def normalize(pts):
    """Make landmarks independent of hand position and distance from camera.
    MUST be identical to the normalize() in train.py."""
    pts = np.asarray(pts, dtype=float).reshape(21, 3)
    pts = pts - pts[0]                           # wrist -> origin
    scale = np.linalg.norm(pts[9]) + 1e-6        # wrist -> middle knuckle
    return (pts / scale).flatten()
 
 
def hand_to_features(hand):
    return normalize([[lm.x, lm.y, lm.z] for lm in hand.landmark])
 
 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
 
cap = cv2.VideoCapture(0)
 
words = []            # completed words, one per line (more if long)
word = ""             # word currently being spelled
last_added_time = time.time()
 
# Voting: accept a letter when VOTE_RATIO of the last HISTORY_LEN frames agree
HISTORY_LEN = 15
VOTE_RATIO = 0.8
MIN_GAP = 0.6         # minimum seconds between added letters
history = deque(maxlen=HISTORY_LEN)
 
MAX_CHARS = 16        # letters per line before wrapping
LINE_HEIGHT = 45
TEXT_START_Y = 130
 
 
def wrap(text, n=MAX_CHARS):
    """Split text into chunks of at most n characters (always at least one chunk)."""
    if not text:
        return [""]
    return [text[i:i + n] for i in range(0, len(text), n)]
 
 
while True:
    success, frame = cap.read()
    if not success:
        break
 
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
 
    prediction = ""
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            prediction = model.predict([hand_to_features(hand)])[0]
 
    history.append(prediction)
    now = time.time()
 
    if len(history) == history.maxlen and now - last_added_time > MIN_GAP:
        letter, n = Counter(history).most_common(1)[0]
        if letter and n / len(history) >= VOTE_RATIO:
            word += letter
            last_added_time = now
            history.clear()
 
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord(' '):
        # Finish the current word and start a new line
        if word:
            words.append(word)
            word = ""
    elif key == 8:  # Backspace
        if word:
            word = word[:-1]
        elif words:
            word = words.pop()   # pull the previous word back to edit it
    elif key == ord('c'):
        words.clear()
        word = ""
    elif key == ord('q'):
        break
 
    # Display
    cv2.putText(frame, f"Sign: {prediction}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
 
    # Build display lines, wrapping long words every MAX_CHARS letters
    lines = []
    for w in words:
        lines.extend(wrap(w))
 
    current = wrap(word)
    if len(current[-1]) >= MAX_CHARS:
        current.append("")      # cursor goes on a fresh line if the last one is full
    current[-1] += "_"          # cursor
    lines.extend(current)
 
    max_lines = max(1, (frame.shape[0] - TEXT_START_Y - 40) // LINE_HEIGHT)
    visible = lines[-max_lines:]  # show the most recent lines if space runs out
 
    for i, line in enumerate(visible):
        y = TEXT_START_Y + i * LINE_HEIGHT
        cv2.putText(frame, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
 
    cv2.putText(frame, "Space=next word  Bksp=delete  C=clear  Q=quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
 
    cv2.imshow("ASL Speller", frame)
 
cap.release()
cv2.destroyAllWindows()
 
