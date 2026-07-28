import cv2
import threading
import numpy as np
import time
import collections
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from deepface import DeepFace

WEBCAM_INDEX    = 0
HISTORY_LENGTH  = 50
MAX_FACES       = 5
SMOOTHING_LEN   = 8
ANALYSIS_SIZE   = 224

EMOTION_COLORS = {
    "Happy":       (0,   220, 80),
    "Sad":         (200, 80,  40),
    "Angry":       (0,   0,   230),
    "Fear":        (130, 0,   200),
    "Surprise":    (0,   180, 230),
    "Disgust":     (0,   130, 0),
    "Neutral":     (160, 160, 160),
    "Excited":     (0,   200, 255),
    "Frustrated":  (0,   60,  200),
    "Panicked":    (80,  0,   220),
    "Anxious":     (130, 80,  180),
    "Bored":       (100, 100, 100),
    "Furious":     (0,   0,   255),
    "Error":       (50,  50,  50),
    "Loading AI...": (0, 200, 0),
    "Ayanakoji":   (100, 0, 100),
}
DEFAULT_COLOR = (0, 200, 0)

_lock              = threading.Lock()
_frames_to_analyze = []
_results           = {}
_emotion_history   = collections.deque(maxlen=HISTORY_LENGTH)
running            = True

_smoothing: dict[str, collections.deque] = {}

def download_model():
    path = "face_landmarker.task"
    if not os.path.exists(path):
        print("Downloading face landmarker model…")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, path)
    return path

def download_hand_model():
    path = "hand_landmarker.task"
    if not os.path.exists(path):
        print("Downloading hand landmarker model…")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, path)
    return path

def _resolve_emotion(em: dict) -> tuple[str, float]:
    """
    Given a smoothed emotion dict (keys lowercase), return (label, confidence).
    Complex emotions take priority when their component scores pass thresholds.
    """
    rules = [
        ("Excited",    em['happy'] > 40 and em['surprise'] > 20,
                       (em['happy'] + em['surprise']) / 1.2),
        ("Furious",    em['angry'] > 50 and em['disgust'] > 20,
                       (em['angry'] + em['disgust']) / 1.2),
        ("Frustrated", em['angry'] > 30 and em['sad']     > 30,
                       (em['angry'] + em['sad'])     / 1.2),
        ("Panicked",   em['surprise'] > 30 and em['fear'] > 30,
                       (em['surprise'] + em['fear']) / 1.2),
        ("Anxious",    em['sad'] > 30 and em['fear']      > 20,
                       (em['sad']  + em['fear'])     / 1.2),
        ("Bored",      em['neutral'] > 85 and em['sad']   > 5,
                       em['neutral']),
    ]
    for label, cond, raw_conf in rules:
        if cond:
            return label, min(99.9, raw_conf)

    best = max(em, key=em.get)
    return best.capitalize(), em[best]

_LEFT_EYE_IDXS  = [33, 133, 159, 145]
_RIGHT_EYE_IDXS = [362, 263, 386, 374]

def _eye_center(landmarks, idxs, img_w, img_h):
    xs = [landmarks[i].x * img_w for i in idxs]
    ys = [landmarks[i].y * img_h for i in idxs]
    return np.array([np.mean(xs), np.mean(ys)])

def align_face(frame, landmarks, out_size=ANALYSIS_SIZE):
    """
    Rotate + crop the face so the eyes are perfectly horizontal.
    This is the single biggest accuracy improvement for emotion models.
    Returns a square BGR image of shape (out_size, out_size).
    """
    h, w = frame.shape[:2]

    lc = _eye_center(landmarks, _LEFT_EYE_IDXS,  w, h)
    rc = _eye_center(landmarks, _RIGHT_EYE_IDXS, w, h)

    dy = rc[1] - lc[1]
    dx = rc[0] - lc[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_mid = (lc + rc) / 2.0

    desired_eye_dist = out_size * 0.35
    actual_eye_dist  = np.linalg.norm(rc - lc)
    scale = desired_eye_dist / max(actual_eye_dist, 1.0)

    M = cv2.getRotationMatrix2D(tuple(eye_mid), angle, scale)

    M[0, 2] += out_size / 2.0 - eye_mid[0]
    M[1, 2] += out_size * 0.45 - eye_mid[1]

    aligned = cv2.warpAffine(frame, M, (out_size, out_size),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned

def preprocess_crop(img):
    """
    CLAHE on the luminance channel so bad / uneven lighting doesn't ruin
    the emotion scores.  Returns BGR.
    """
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def analyze_emotion_thread():
    global running, _results, _emotion_history

    while running:
        with _lock:
            batch = list(_frames_to_analyze)

        if not batch:
            time.sleep(0.05)
            continue

        new_results = {}

        for crop, face_id in batch:
            try:
                processed = preprocess_crop(crop)
                small = cv2.resize(processed, (ANALYSIS_SIZE, ANALYSIS_SIZE))

                result = DeepFace.analyze(
                    small,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]

                raw = result['emotion']

                if face_id not in _smoothing:
                    _smoothing[face_id] = collections.deque(maxlen=SMOOTHING_LEN)
                _smoothing[face_id].append(raw)

                q = _smoothing[face_id]
                smoothed = {k: sum(f[k] for f in q) / len(q) for k in raw}

                label, conf = _resolve_emotion(smoothed)
                new_results[face_id] = (label, conf)

            except Exception:
                new_results[face_id] = ("Error", 0.0)

        with _lock:
            _results = new_results
            primary_id = batch[0][1] if batch else None
            if primary_id and primary_id in new_results:
                _emotion_history.append(new_results[primary_id])

class FaceTracker:
    """
    Assigns stable string IDs to faces across frames so smoothing queues
    don't get mixed up when face count changes or faces swap positions.
    """
    def __init__(self, max_dist=80):
        self.tracks: dict[str, tuple[int, int]] = {}
        self.next_id = 1
        self.max_dist = max_dist

    def update(self, centers: list[tuple[int, int]]) -> list[str]:
        """Match centers to existing tracks; return list of IDs."""
        unmatched_tracks = set(self.tracks.keys())
        assigned = {}

        for cx, cy in centers:
            best_id, best_d = None, float('inf')
            for tid, (tx, ty) in self.tracks.items():
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d

            if best_id and best_d < self.max_dist:
                assigned[(cx, cy)] = best_id
                self.tracks[best_id] = (cx, cy)
                unmatched_tracks.discard(best_id)
            else:
                new_id = f"F{self.next_id}"
                self.next_id += 1
                self.tracks[new_id] = (cx, cy)
                assigned[(cx, cy)] = new_id

        for tid in unmatched_tracks:
            del self.tracks[tid]
            _smoothing.pop(tid, None)

        return [assigned[(cx, cy)] for cx, cy in centers]

def draw_graph(frame, history, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 15, 0), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 1)
    cv2.putText(frame, "FACE 1 CONFIDENCE", (x + 5, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 0), 1)

    if len(history) < 2:
        return

    pts = []
    for i, (_, conf) in enumerate(history):
        px = int(x + (i / HISTORY_LENGTH) * w)
        py = int(y + h - (conf / 100.0) * (h - 20))
        pts.append((px, py))

    arr = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [arr], isClosed=False, color=(0, 220, 80), thickness=2)

    if history:
        last_label, last_conf = history[-1]
        color = EMOTION_COLORS.get(last_label, DEFAULT_COLOR)
        cv2.putText(frame, f"{last_label} {last_conf:.0f}%",
                    (x + 4, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)

def main():
    global running, _frames_to_analyze

    model_path = download_model()

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=MAX_FACES,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    hand_model_path = download_hand_model()
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    tracker = FaceTracker()

    t = threading.Thread(target=analyze_emotion_thread, daemon=True)
    t.start()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        running = False
        return

    print("Running. Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = detector.detect(mp_img)
        hand_detection = hand_detector.detect(mp_img)
        
        hand_centers = []
        if hand_detection.hand_landmarks:
            for hand_lms in hand_detection.hand_landmarks:
                xs = [lm.x * w for lm in hand_lms]
                ys = [lm.y * h for lm in hand_lms]
                hand_centers.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))

        centers, boxes, crops = [], [], []

        if detection.face_landmarks:
            for face_lms in detection.face_landmarks:
                xs = [lm.x * w for lm in face_lms]
                ys = [lm.y * h for lm in face_lms]
                xmin, xmax = int(min(xs)), int(max(xs))
                ymin, ymax = int(min(ys)), int(max(ys))

                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2
                centers.append((cx, cy))
                boxes.append((xmin, ymin, xmax, ymax))

                try:
                    crop = align_face(frame, face_lms, out_size=ANALYSIS_SIZE)
                except Exception:
                    size = max(xmax - xmin, ymax - ymin)
                    pad  = int(size * 0.25)
                    size += pad * 2
                    y1   = max(0, cy - size // 2)
                    y2   = min(h, cy + size // 2)
                    x1   = max(0, cx - size // 2)
                    x2   = min(w, cx + size // 2)
                    crop = frame[y1:y2, x1:x2]
                crops.append(crop if crop.size > 0 else None)

        face_ids = tracker.update(centers)

        new_batch = [
            (crop, fid)
            for crop, fid in zip(crops, face_ids)
            if crop is not None
        ]
        with _lock:
            _frames_to_analyze = new_batch

        with _lock:
            snap = dict(_results)

        for i, (fid, (xmin, ymin, xmax, ymax)) in enumerate(zip(face_ids, boxes)):
            label, conf = snap.get(fid, ("Loading AI...", 0.0))
            
            for hcx, hcy in hand_centers:
                if xmin < hcx < xmax and ymin < hcy < ymax:
                    label = "Ayanakoji"
                    conf = 100.0
                    break

            color = EMOTION_COLORS.get(label, DEFAULT_COLOR)

            cv2.rectangle(frame,
                          (xmin - 20, ymin - 30),
                          (xmax + 20, ymax + 20),
                          color, 2)

            display_num = i + 1
            text = f"Face {display_num}: {label}"
            if label not in ("Error", "Loading AI..."):
                text += f" ({conf:.1f}%)"

            cv2.putText(frame, text,
                        (xmin - 20, ymin - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with _lock:
            hist = list(_emotion_history)
        draw_graph(frame, hist, w - 220, h - 120, 200, 100)
        draw_fps(frame, fps)

        cv2.imshow("Emotion HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()