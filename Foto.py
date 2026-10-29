import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model disimpan di:
# D:\VSCode\PY\media\hand_landmarker.task
MODEL_DIR = os.path.join(BASE_DIR, "media")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/1/hand_landmarker.task"
)


# ============================================================
# DOWNLOAD MODEL JIKA BELUM ADA
# ============================================================

def ensure_model():
    if os.path.isfile(MODEL_PATH):
        print("Model ditemukan:")
        print(MODEL_PATH)
        return True

    print("Model hand_landmarker.task tidak ditemukan.")
    print("Mencoba mengunduh model...")

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        if os.path.isfile(MODEL_PATH):
            print("Model berhasil diunduh:")
            print(MODEL_PATH)
            return True

    except Exception as e:  # noqa: BLE001
        print("Gagal mengunduh model:")
        print(e)

    return False


# ============================================================
# DETEKSI JARI
# ============================================================

def fingers_up(hand_landmarks, handedness_label):
    lm = hand_landmarks

    fingers = []

    # Thumb
    if handedness_label == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # Index, Middle, Ring, Pinky
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    for tip, pip in zip(tip_ids, pip_ids):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers


# ============================================================
# PEACE SIGN
# ============================================================

def is_peace_sign(fingers):
    _thumb, index, middle, ring, pinky = fingers

    return (
        index
        and middle
        and not ring
        and not pinky
    )


# ============================================================
# CARI KAMERA
# ============================================================

def find_camera():
    print("Mencari kamera...")

    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret and frame is not None and frame.mean() > 1:
                cap.release()
                print(f"Kamera ditemukan: {i}")
                return i

            cap.release()

    return 0


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Pastikan model ada
    # --------------------------------------------------------

    if not ensure_model():
        print()
        print("MODEL TIDAK DITEMUKAN.")
        print("Pastikan file:")
        print(MODEL_PATH)
        print("tersedia.")
        return

    # --------------------------------------------------------
    # Kamera
    # --------------------------------------------------------

    cam_idx = find_camera()

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Tidak bisa membuka kamera.")
        return

    # Resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print()
    print(f"Kamera {cam_idx} berhasil dibuka.")
    print("Angkat tangan membentuk peace (V) untuk mengambil foto.")
    print("Tekan Q untuk keluar.")
    print()

    # --------------------------------------------------------
    # MediaPipe Hand Landmarker
    # --------------------------------------------------------

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=4,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    try:
        landmarker = HandLandmarker.create_from_options(options)
    except Exception as e:  # noqa: BLE001
        print()
        print("Gagal membuat HandLandmarker.")
        print(e)
        cap.release()
        return

    # Timestamp harus selalu meningkat
    start_time = time.monotonic()
    
    flash_counter = 0
    last_peace_detected = False
    last_capture_time = 0.0
    CAPTURE_COOLDOWN = 2.0  # Jeda minimal antar foto (detik)

    try:

        while True:

            # ------------------------------------------------
            # Baca kamera
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                print("Gagal membaca frame.")
                break

            # Mirror kamera
            frame = cv2.flip(frame, 1)

            # ------------------------------------------------
            # Convert BGR -> RGB
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Timestamp dalam milliseconds
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            # ------------------------------------------------
            # Deteksi tangan
            # ------------------------------------------------

            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            peace_detected = False

            # ------------------------------------------------
            # Proses hasil
            # ------------------------------------------------

            if results.hand_landmarks and results.handedness:

                for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness
                ):

                    label = handedness[0].category_name

                    fingers = fingers_up(
                        hand_landmarks,
                        label
                    )

                    if is_peace_sign(fingers):
                        peace_detected = True

            # ------------------------------------------------
            # PEACE SIGN -> FLASH
            # ------------------------------------------------

            # Debounce: Pastikan peace benar-benar hilang sebelum bisa capture lagi
            if peace_detected:
                now = time.monotonic()

                if (
                    not last_peace_detected
                    and (now - last_capture_time) > CAPTURE_COOLDOWN
                ):
                    flash_counter = 7  # Durasi flash

                    # Simpan foto bersih (tanpa landmark / teks)
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    ms_str = f"{int((now % 1) * 1000):03d}"
                    filename = f"capture_{timestamp_str}_{ms_str}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Foto tersimpan: {filename}")

                    last_capture_time = now

                last_peace_detected = True
            else:
                last_peace_detected = False

            # ------------------------------------------------
            # Gambar landmark tangan (untuk preview saja)
            # ------------------------------------------------

            if results.hand_landmarks:

                for hand_landmarks in results.hand_landmarks:
                    draw_landmarks(
                        frame,
                        hand_landmarks,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                    )

            if flash_counter > 0:
                # Efek flash putih
                white_overlay = np.full(frame.shape, 255, dtype=np.uint8)
                # Alpha mengecil seiring waktu untuk efek fade out
                alpha = flash_counter / 7.0
                frame = cv2.addWeighted(frame, 1 - alpha, white_overlay, alpha, 0)
                flash_counter -= 1

                cv2.putText(
                    frame,
                    "CHEESE!",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            else:

                cv2.putText(
                    frame,
                    "Show PEACE SIGN to take photo",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # ------------------------------------------------
            # Tampilkan
            # ------------------------------------------------

            cv2.imshow(
                "Camera Flash Capture",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Keluar...")
                break

    except Exception as e:  # noqa: BLE001

        print()
        print("Terjadi error:")
        print(e)

        import traceback
        traceback.print_exc()

    finally:

        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()