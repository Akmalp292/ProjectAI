import os
import pickle
import cv2
import mediapipe as mp
import albumentations as A

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Definisikan direktori dataset
data_dir = './ASL dataset'
dataset = []
labels = []

# Pipeline augmentasi dengan Albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.5),
])

# Loop untuk setiap folder kelas
for directory in os.listdir(data_dir):
    path = os.path.join(data_dir, directory)

    for img_path in os.listdir(path):
        # Load gambar asli
        image_path = os.path.join(path, img_path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Unable to load image at path: {image_path}")
            continue

        # Terapkan augmentasi ke gambar asli
        augmented = transform(image=image)
        image_aug = augmented['image']

        # Proses gambar augmented ke RGB untuk MediaPipe
        image_rgb = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
        processed_image = hands.process(image_rgb)

        hand_landmarks = processed_image.multi_hand_landmarks

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                normalized_landmarks = []
                x_coordinates = [lm.x for lm in hand_landmark.landmark]
                y_coordinates = [lm.y for lm in hand_landmark.landmark]

                min_x, min_y = min(x_coordinates), min(y_coordinates)

                # Normalisasi koordinat landmark
                for lm in hand_landmark.landmark:
                    normalized_x = lm.x - min_x
                    normalized_y = lm.y - min_y
                    normalized_landmarks.extend([normalized_x, normalized_y])

                # Tambahkan fitur dan label ke dataset
                dataset.append(normalized_landmarks)
                labels.append(directory)

# Simpan dataset dan label ke file pickle
with open("./ASL_augmented.pickle", "wb") as f:
    pickle.dump({"dataset": dataset, "labels": labels}, f)

print("Dataset dengan augmentasi berhasil dibuat dan disimpan ke ASL_augmented.pickle")
