import cv2
import mediapipe as mp
import pyautogui
import math
import urllib.request
import os

# ── Setup ──────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen_w, screen_h = pyautogui.size()

# ── Margin zone (tweak these if cursor doesn't reach corners) ──────────
MARGIN_LEFT   = 0.05
MARGIN_RIGHT  = 0.95
MARGIN_TOP    = 0.05
MARGIN_BOTTOM = 0.95

# ── Smoothing (increase if cursor shakes, decrease if it feels slow) ───
SMOOTHING = 15

# ── New MediaPipe API (0.10.30+) ───────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~3MB), please wait...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded! Starting...")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── State ──────────────────────────────────────────────────────────────
smooth_x, smooth_y = 0, 0
clicking = False

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def draw_landmarks(frame, landmarks, w, h):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)

print("Air Mouse running. Press Q to quit.")
print("Controls:")
print("  Move index finger  → move cursor")
print("  Pinch index+thumb  → left click")
print("  Pinch index+middle → right click")

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            index_tip  = lm[8]
            thumb_tip  = lm[4]
            middle_tip = lm[12]

            # ── Move cursor with margin zone fix ──────────────────────
            clamped_x = max(MARGIN_LEFT,  min(MARGIN_RIGHT,  index_tip.x))
            clamped_y = max(MARGIN_TOP,   min(MARGIN_BOTTOM, index_tip.y))

            mapped_x = (clamped_x - MARGIN_LEFT)  / (MARGIN_RIGHT  - MARGIN_LEFT)  * screen_w
            mapped_y = (clamped_y - MARGIN_TOP)    / (MARGIN_BOTTOM - MARGIN_TOP)   * screen_h

            smooth_x = smooth_x + (mapped_x - smooth_x) / SMOOTHING
            smooth_y = smooth_y + (mapped_y - smooth_y) / SMOOTHING
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # ── Left click: pinch index + thumb ───────────────────────
            pinch_dist = distance(index_tip, thumb_tip)
            if pinch_dist < 0.05:
                if not clicking:
                    pyautogui.click()
                    clicking = True
                    cv2.putText(frame, "LEFT CLICK", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                clicking = False

            # ── Right click: pinch index + middle finger ──────────────
            if distance(index_tip, middle_tip) < 0.04:
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            draw_landmarks(frame, lm, w, h)

            # ── HUD ───────────────────────────────────────────────────
            cv2.putText(frame, f"Cursor: ({int(smooth_x)}, {int(smooth_y)})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Draw active zone rectangle on camera feed
            ax1 = int(MARGIN_LEFT   * w)
            ay1 = int(MARGIN_TOP    * h)
            ax2 = int(MARGIN_RIGHT  * w)
            ay2 = int(MARGIN_BOTTOM * h)
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (100, 100, 255), 1)
            cv2.putText(frame, "Active Zone", (ax1 + 4, ay1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)

        else:
            cv2.putText(frame, "Show your hand to the camera",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Air Mouse - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Air Mouse stopped.")