# AntiSpoofV7.py - Enhanced Anti Spoof + DeepFake Detection
import cv2
import numpy as np
from scipy.fftpack import fft2
from collections import deque

class AntiSpoofV7:
    def __init__(self,
                 face_size=(224,224),
                 buffer_size=10,
                 blink_thr=0.15,
                 motion_thr=0.08,
                 fft_thr=0.12,
                 moire_thr=15.0,
                 spoof_score_limit=0.55):

        self.face_size = face_size
        self.buffer_size = buffer_size
        self.blink_thr = blink_thr
        self.motion_thr = motion_thr
        self.fft_thr = fft_thr
        self.moire_thr = moire_thr
        self.spoof_score_limit = spoof_score_limit

        # Buffers for temporal smoothing
        self.motion_buffer = deque(maxlen=buffer_size)
        self.fft_buffer = deque(maxlen=buffer_size)
        self.moire_buffer = deque(maxlen=buffer_size)
        self.blink_buffer = deque(maxlen=buffer_size)

        self.last_frames = deque(maxlen=buffer_size)
        self.last_blink_values = deque(maxlen=buffer_size)

    # ---------------- Motion ----------------
    def detect_motion(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        self.last_frames.append(gray)
        if len(self.last_frames) < 2:
            mot = 1.0
        else:
            diff = cv2.absdiff(self.last_frames[-1], self.last_frames[-2])
            mot = np.mean(diff) / 25.0
        self.motion_buffer.append(mot)
        return np.mean(self.motion_buffer)

    # ---------------- Moiré ----------------
    def detect_moire(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        F = np.log(np.abs(np.fft.fftshift(fft2(gray))) + 1)
        h, w = F.shape
        center = F[h//2-10:h//2+10, w//2-10:w//2+10]
        high_freq = np.mean(F) - np.mean(center)
        self.moire_buffer.append(high_freq)
        return np.mean(self.moire_buffer)

    # ---------------- Blink ----------------
    def detect_blink(self, face_landmarks):
        try:
            eye_left = face_landmarks[36:42]
            eye_right = face_landmarks[42:48]

            def EAR(eye):
                A = np.linalg.norm(eye[1] - eye[5])
                B = np.linalg.norm(eye[2] - eye[4])
                C = np.linalg.norm(eye[0] - eye[3])
                return (A + B) / (2.0 * C)

            ear_left = EAR(np.array(eye_left))
            ear_right = EAR(np.array(eye_right))
            ear = (ear_left + ear_right) / 2.0
            self.last_blink_values.append(ear)
            self.blink_buffer.append(ear)
            return np.mean(self.blink_buffer)
        except:
            return 0.3

    # ---------------- FFT Artifact ----------------
    def detect_fft_artifact(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        F = np.log(np.abs(np.fft.fftshift(fft2(gray))) + 1)
        h, w = F.shape
        center = F[h//2-15:h//2+15, w//2-15:w//2+15]
        energy_high = np.sum(F) - np.sum(center)
        energy_total = np.sum(F)
        ratio = energy_high / (energy_total + 1e-6)
        self.fft_buffer.append(ratio)
        return np.mean(self.fft_buffer)

    # ---------------- LBP Texture ----------------
    def detect_lbp_texture(self, face_crop):
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        lbp = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # đơn giản: chuẩn hóa var của LBP
        var = np.var(lbp) / 255.0
        return var

    # ---------------- Check Frame ----------------
    def check_frame(self, frame, bbox, landmarks=None, face_id=None):
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            return False, 0.0

        face_crop = cv2.resize(face_crop, self.face_size)

        # compute signals
        motion = self.detect_motion(face_crop)
        moire = self.detect_moire(face_crop)
        fft_ratio = self.detect_fft_artifact(face_crop)
        blink = 0.2
        if landmarks is not None:
            blink = self.detect_blink(landmarks)
        lbp = self.detect_lbp_texture(face_crop)

        # Combined score
        score = (
            0.25 * min(motion,1.0) +
            0.20 * (1 - min(moire/40, 1.0)) +
            0.25 * min(blink*3,1.0) +
            0.20 * (1 - min(fft_ratio*3,1.0)) +
            0.10 * min(lbp,1.0)
        )

        is_real = score >= self.spoof_score_limit
        return is_real, float(score)
