import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

# Veri yolu ve sınıf adları
actions = np.array(os.listdir("keypoints"))  # "keypoints" alt klasörlerinden sınıf adlarını al
label_map = {label: num for num, label in enumerate(actions)}

# Veri işleme
DATA_PATH = "keypoints"
sequences, labels = [], []

for action in actions:  # Her bir sınıf için
    action_path = os.path.join(DATA_PATH, action)
    for video_folder in os.listdir(action_path):  # Her bir video için
        video_path = os.path.join(action_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        window = []
        frame_files = sorted([f for f in os.listdir(video_path) if f.startswith("frame_") and f.endswith(".npy")], key=lambda x: int(x.split('_')[1].split('.')[0]))
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            res = np.load(frame_path)
            window.append(res)

        if len(window) > 0:  # En az bir frame varsa sekansı ekle
            sequences.append(window)
            labels.append(label_map[action])

# Numpy dizilerine dönüştürme ve padding
max_sequence_length = max(len(seq) for seq in sequences)
X = np.array([np.pad(seq, ((0, max_sequence_length - len(seq)), (0, 0)), mode='constant') for seq in sequences])
y = to_categorical(labels).astype(int)

# Eğitim ve test veri setlerinin ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Model oluşturma
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(max_sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Modeli derleme ve eğitme
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

# Model özeti
#model.summary()
model.save('actions.keras') #.keras uzantısını kendisi önerdi neden bilmiyorum
