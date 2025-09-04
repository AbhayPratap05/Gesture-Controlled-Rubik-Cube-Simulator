import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from cube import Cube2D
import math

# ---------------- Mediapipe Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Camera Setup ----------------
cap = cv2.VideoCapture(0)

# ---------------- Gesture History & Movement Tracking ----------------
gesture_history = deque(maxlen=10)  # increased for better smoothing
position_history = deque(maxlen=10)  # track finger positions for swipe detection
last_gesture_time = 0
gesture_cooldown = 0.5  # seconds between gestures

# ---------------- Cube Setup ----------------
cube = Cube2D()
selected_row = 1
selected_col = 1
selected_face = 'F'

# ---------------- Functions ----------------
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (different logic for left/right hand)
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def get_distance(p1, p2):
    """Calculate distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_pinch(hand_landmarks):
    """Detect pinch gesture (thumb and index finger close)"""
    thumb_tip = [hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y]
    index_tip = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]
    distance = get_distance(thumb_tip, index_tip)
    return distance < 0.05  # threshold for pinch

def detect_swipe_from_history():
    if len(position_history) < 5:
        return None
    start_pos = position_history[0]
    end_pos = position_history[-1]
    dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]

    # Scale threshold by movement range
    movement_magnitude = math.hypot(dx, dy)
    if movement_magnitude < 0.05:  # smaller threshold for closer hand
        return None

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

def classify_gesture(hand_landmarks):
    """Classify the current gesture"""
    fingers = fingers_up(hand_landmarks)
    finger_count = sum(fingers)
    
    # Get current position of index finger for movement tracking
    index_pos = [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y]
    position_history.append(index_pos)
    
    # Fist
    if finger_count == 0:
        return "FIST"
    
    # Palm (all fingers up)
    elif finger_count == 5:
        return "PALM"
    
    # Pinch gesture
    elif detect_pinch(hand_landmarks):
        return "PINCH"
    
    # Two fingers (index + middle)
    elif fingers[1] == 1 and fingers[2] == 1 and finger_count == 2:
        swipe = detect_swipe_from_history()
        if swipe == "UP":
            return "FACE_CCW"
        elif swipe == "DOWN":
            return "FACE_CW"
        elif swipe in ["LEFT", "RIGHT"]:
            return f"TWO_FINGER_{swipe}"
        else:
            return "TWO_FINGERS"
    
    # Single index finger
    elif fingers[1] == 1 and finger_count == 1:
        swipe = detect_swipe_from_history()
        if swipe:
            return f"SWIPE_{swipe}"
        else:
            return "POINT"
    
    return "UNKNOWN"

def execute_gesture(gesture):
    """Execute the cube operation based on gesture"""
    global selected_row, selected_col, selected_face
    
    if gesture == "SWIPE_LEFT":
        cube.rotate_row_left(selected_row)
        print(f"Row {selected_row} rotated left")
    
    elif gesture == "SWIPE_RIGHT":
        cube.rotate_row_right(selected_row)
        print(f"Row {selected_row} rotated right")
    
    elif gesture == "SWIPE_UP":
        cube.rotate_col_up(selected_col)
        print(f"Column {selected_col} rotated up")
    
    elif gesture == "SWIPE_DOWN":
        cube.rotate_col_down(selected_col)
        print(f"Column {selected_col} rotated down")
    
    elif gesture == "FACE_CW":
        cube.rotate_face_cw(selected_face)
        print(f"Face {selected_face} rotated clockwise")
    
    elif gesture == "FACE_CCW":
        cube.rotate_face_ccw(selected_face)
        print(f"Face {selected_face} rotated counter-clockwise")
    
    # elif gesture == "FIST":
    #     cube.reset()
    #     print("Cube reset")
    
    elif gesture == "PINCH":
        # Cycle through selection modes
        selected_row = (selected_row + 1) % 3
        selected_col = (selected_col + 1) % 3
        faces = ['U', 'R', 'F', 'D', 'L', 'B']
        current_face_idx = faces.index(selected_face)
        selected_face = faces[(current_face_idx + 1) % len(faces)]
        print(f"Selected: Row {selected_row}, Col {selected_col}, Face {selected_face}")

def draw_selection_indicator(img, selected_row, selected_col):
    """Draw indicators on the cube image to show current selection"""
    size = 50
    # Highlight selected row and column on the front face
    face_x, face_y = 3 * size, 3 * size  # Front face position
    
    # Highlight selected row
    cv2.rectangle(img, 
                  (face_x, face_y + selected_row * size),
                  (face_x + 3 * size, face_y + (selected_row + 1) * size),
                  (255, 255, 0), 3)
    
    # Highlight selected column
    cv2.rectangle(img, 
                  (face_x + selected_col * size, face_y),
                  (face_x + (selected_col + 1) * size, face_y + 3 * size),
                  (0, 255, 255), 3)

# ---------------- Main Loop ----------------
import time

palm_hold_start = None
palm_hold_duration = 2.5

print("Gesture Controls:")
print("- Single finger swipe: Move rows/columns")
print("- Two fingers up/down: Rotate face")
print("- Palm: Hold to Reset cube")
print("- Pinch: Change selection")
print("- Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()
    current_gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Classify gesture
            gesture = classify_gesture(hand_landmarks)
            gesture_history.append(gesture)
            
            # Use majority vote for gesture smoothing
            if len(gesture_history) >= 5:
                # Count occurrences of each gesture
                gesture_counts = {}
                for g in list(gesture_history)[-5:]:  # Last 5 gestures
                    gesture_counts[g] = gesture_counts.get(g, 0) + 1
                
                # Get most common gesture
                current_gesture = max(gesture_counts, key=gesture_counts.get)

                # Add palm hold detection here
                if current_gesture == "PALM":
                    if palm_hold_start is None:
                        palm_hold_start = current_time
                    elif current_time - palm_hold_start > palm_hold_duration:
                        cube.reset()
                        print("Cube reset from palm hold")
                        palm_hold_start = None
                else:
                    palm_hold_start = None
                
                # Execute gesture if it's action-based and cooldown has passed
                if (current_gesture in ["SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN", 
                                       "FACE_CW", "FACE_CCW", "FIST", "PINCH"] and 
                    current_time - last_gesture_time > gesture_cooldown):
                    
                    execute_gesture(current_gesture)
                    last_gesture_time = current_time
                    
                    # Clear history after executing gesture to prevent repeats
                    gesture_history.clear()
                    position_history.clear()

    else:
        # Clear history when no hand is detected
        gesture_history.clear()
        position_history.clear()

    # ---------------- Draw Cube with Selection Indicators ----------------
    cube_img = cube.draw(size=50)
    draw_selection_indicator(cube_img, selected_row, selected_col)
    cv2.imshow("2D Cube", cube_img)

    # ---------------- Show Current Status ----------------
    info_text = [
        f"Gesture: {current_gesture if current_gesture else 'None'}",
        f"Selected Row: {selected_row}, Col: {selected_col}, Face: {selected_face}"
    ]
    
    # Show palm hold progress
    if palm_hold_start is not None:
        hold_time = current_time - palm_hold_start
        progress = min(hold_time / palm_hold_duration, 1.0)
        info_text.append(f"Reset progress: {progress:.1%}")
    
    info_text.append("ESC to exit")
    
    for i, text in enumerate(info_text):
        color = (0, 255, 0)  # Green
        if i == 2 and palm_hold_start is not None:  # Progress text in red
            color = (0, 0, 255)
        cv2.putText(frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()