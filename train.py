import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from scipy.interpolate import CubicSpline
from sklearn.model_selection import KFold

# Define data path and load actions
DATA_PATH = "keypoints"
actions = np.array([
    d for d in os.listdir(DATA_PATH) 
    if os.path.isdir(os.path.join(DATA_PATH, d))
])
label_map = {label: num for num, label in enumerate(actions)}


def fill_missing_keypoints(sequence):

    sequence = np.array(sequence)

    for i in range(sequence.shape[1]):  # Iterate over each column (x, y, z)
        column = sequence[:, i]
        mask_column = column == 0  # Create a mask for missing values (zeros)

        if np.any(~mask_column):  # Check if there is at least one known value
            known_x = np.flatnonzero(~mask_column)  # Indices of known points
            known_y = column[~mask_column]  # Values of known points

            # Create cubic spline interpolation function using known points
            cs = CubicSpline(known_x, known_y, bc_type='natural')

            missing_x = np.flatnonzero(mask_column)  # Indices of missing points
            column[mask_column] = cs(missing_x)  # Estimate and fill missing points

        sequence[:, i] = column  # Save the updated column back to sequence

    return sequence 

# Data processing
sequences, labels = [], []
# before_fill_val = [] # just for check
for action in actions:  # Iterate over each class
    action_path = os.path.join(DATA_PATH, action)
    for video_folder in os.listdir(action_path):  # Each video
        video_path = os.path.join(action_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        window = []
        frame_files = sorted(
            [f for f in os.listdir(video_path) if f.startswith("frame_") and f.endswith(".npy")],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            res = np.load(frame_path)
            window.append(res)

        # window'a interpolation uygula
        filled_miss_val_window = fill_missing_keypoints(window)

        if len(window) > 0:  # Add sequence if it has frames
            # before_fill_val.append(window) # !!just for check
            sequences.append(filled_miss_val_window)
            labels.append(label_map[action])


# Convert to NumPy arrays and normalize input data
X = np.array(sequences)
# X = (X - X.min()) / (X.max() - X.min())  # Normalize data
y = to_categorical(labels).astype(int)



k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
all_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\n🔹 Training on Fold {fold+1}/{k}...")

    # Eğitim ve doğrulama setlerini oluştur
    X_train_fold, X_val_fold = X[train_index], X[test_index]
    y_train_fold, y_val_fold = y[train_index], y[test_index]


    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  
    model.add(Dropout(0.3))  

    model.add(LSTM(64, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  
    model.add(Dropout(0.3))  

    model.add(LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='tanh',kernel_regularizer=l2(0.01)))
    model.add(Dense(32, activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0005),
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])


    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )

    log_dir = os.path.join('Logs', f"fold_{fold+1}")
    tb_callback = TensorBoard(log_dir=log_dir)

    model.fit(X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=200, batch_size=32,
            callbacks=[tb_callback, early_stopping, reduce_lr],
            verbose=1)



    # Fold'un Doğruluk Skorunu Kaydet**
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    all_scores.append(val_acc)  
    print(f"✔ Fold {fold+1} Validation Accuracy: {val_acc * 100:.2f}%")

# Ortalama Sonuçları Hesapla**
mean_accuracy = np.mean(all_scores)
print(f"\n K-Fold Cross Validation Sonucu: Ortalama Doğruluk = {mean_accuracy * 100:.2f}%")

# Modeli Kaydet (Son Eğitilen Model)**
model.save('actions.h5')

# Test Seti ile Modeli Değerlendir**
yhat = model.predict(X_val_fold)
ytrue = np.argmax(y_val_fold, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# **Doğruluk ve Confusion Matrix Hesapla**
accuracy = accuracy_score(ytrue, yhat)
print(f"\nModel Accuracy (Son Fold): {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(ytrue, yhat)
print("\nConfusion Matrix:")
print(conf_matrix)