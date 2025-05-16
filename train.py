import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization, Bidirectional, LayerNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from scipy.interpolate import CubicSpline
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, Concatenate


# Define data path and load actions
DATA_PATH = "keypoints_24"
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

        print(f"video_path: {video_path}")

        # window'a interpolation uygula
        # filled_miss_val_window = fill_missing_keypoints(window)
        filled_miss_val_window = window

        if len(window) > 0:  # Add sequence if it has frames
            # before_fill_val.append(window) # !!just for check
            sequences.append(filled_miss_val_window)
            labels.append(label_map[action])


# Convert to NumPy arrays and normalize input data
X = np.array(sequences)

X_r_keypoint = X[:, :, 162:225]
X_r_wrist = X[:, :, 229:230]
X_r_angle = X[:, :, 231:246]
X_r_interaction = X[:, :, 261:277]

X_right = np.concatenate((X_r_keypoint, X_r_wrist, X_r_angle, X_r_interaction), axis=2)
print(f"X_right.shape: {X_right.shape}")

#######################################################################################

X_l_keypoint = X[:, :, 99:162]
X_l_wrist = X[:, :, 230:231]
X_l_angle = X[:, :, 246:261]
X_l_interaction = X[:, :, 277:293]

X_left = np.concatenate((X_l_keypoint, X_l_wrist, X_l_angle, X_l_interaction), axis=2)
print(f"X_right.shape: {X_right.shape}")

#######################################################################################

X_p_keypoint = X[:, :, 0:99]
X_p_angle = X[:, :, 225:229]
X_p_distance = X[:, :, 299:307]
X_p_intra_hand = X[:, :, 293:299]

X_pose = np.concatenate((X_p_keypoint, X_p_angle, X_p_distance, X_p_intra_hand), axis=2)
print(f"X_pose.shape: {X_pose.shape}")
#######################################################################################


# X = (X - X.min()) / (X.max() - X.min())  # Normalize data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

num_samples, timesteps, num_features = X.shape
print(f"num_samples: {num_samples}")
print(f"timesteps: {timesteps}")
print(f"num_features: {num_features}")

# (samples * timesteps, features) şekline getirip
# X_reshaped = X.reshape(-1, num_features)

# # Her feature için ayrı min/max bul
# X_min = X_reshaped.min(axis=0)
# X_max = X_reshaped.max(axis=0)

# # Normalize et
# X_norm = (X_reshaped - X_min) / (X_max - X_min + 1e-8)

# # Eski şekle döndür
# X = X_norm.reshape(num_samples, timesteps, num_features)

y = to_categorical(labels).astype(int)




k = 5
# kf = KFold(n_splits=k, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

all_scores = []


best_acc = 0
best_model = None

# for fold, (train_index, test_index) in enumerate(kf.split(X)):
for fold, (train_index, test_index) in enumerate(skf.split(X_right, labels)):
        
    print(f"\n🔹 Training on Fold {fold+1}/{k}...")

    # Eğitim ve doğrulama setlerini oluştur
    X_right_train, X_right_val = X_right[train_index], X_right[test_index]
    X_left_train, X_left_val = X_left[train_index], X_left[test_index]
    X_pose_train, X_pose_val = X_pose[train_index], X_pose[test_index]
    y_train_fold, y_val_fold = y[train_index], y[test_index]

    # --- Inputs ---
    input_right = Input(shape=(X_right.shape[1], X_right.shape[2]))
    input_left = Input(shape=(X_left.shape[1], X_left.shape[2]))
    input_pose = Input(shape=(X_pose.shape[1], X_pose.shape[2]))

    # --- Sağ el LSTM pipeline ---
    x_r = LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01))(input_right)
    x_r = LayerNormalization(axis=1)(x_r)
    x_r = Dropout(0.4)(x_r)

    # --- Sol el LSTM pipeline ---
    x_l = LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01))(input_left)
    x_l = LayerNormalization(axis=1)(x_l)
    x_l = Dropout(0.4)(x_l)

    # --- Pose LSTM pipeline ---
    x_p = LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01))(input_pose)
    x_p = LayerNormalization(axis=1)(x_p)
    x_p = Dropout(0.4)(x_p)

    # --- Hepsini Attention ile birleştir ---
    merged = Concatenate(axis=2)([x_r, x_l, x_p])

    attn_small = MultiHeadAttention(num_heads=4, key_dim=32)(merged, merged)
    attn_large = MultiHeadAttention(num_heads=8, key_dim=64)(merged, merged)
    attn_out = Add()([attn_small, attn_large])
    attn_out = LayerNormalization(axis=1)(attn_out)

    # --- Global pooling ---
    x = GlobalAveragePooling1D()(attn_out)

    # --- Dense layers ---
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    # --- Output ---
    output_layer = Dense(len(actions), activation='softmax')(x)

    # Model
    model = Model(inputs=[input_right, input_left, input_pose], outputs=output_layer)


    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])



    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=30, restore_best_weights=True
    )

    log_dir = os.path.join('Logs', f"fold_{fold+1}")
    tb_callback = TensorBoard(log_dir=log_dir)

    model.fit(
        [X_right[train_index], X_left[train_index], X_pose[train_index]],
        y_train_fold,
        validation_data=(
            [X_right[test_index], X_left[test_index], X_pose[test_index]],
            y_val_fold
        ),
        epochs=100,
        batch_size=64,
        callbacks=[tb_callback, early_stopping, reduce_lr],
        verbose=1
    )




    # Fold'un Doğruluk Skorunu Kaydet**
    val_loss, val_acc = model.evaluate([X_right_val, X_left_val, X_pose_val], y_val_fold, verbose=0)
    all_scores.append(val_acc)  
    print(f"✔ Fold {fold+1} Validation Accuracy: {val_acc * 100:.2f}%")

    # En iyi modeli kaydet
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model

# Ortalama Sonuçları Hesapla**
mean_accuracy = np.mean(all_scores)
print(f"\n K-Fold Cross Validation Sonucu: Ortalama Doğruluk = {mean_accuracy * 100:.2f}%")

# En iyi modeli kaydet
best_model.save('action2.h5')

# # Modeli Kaydet (Son Eğitilen Model)**
# model.save('actions.h5')

# Test Seti ile Modeli Değerlendir**
yhat = model.predict([X_right_val, X_left_val, X_pose_val])
ytrue = np.argmax(y_val_fold, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# **Doğruluk ve Confusion Matrix Hesapla**
accuracy = accuracy_score(ytrue, yhat)
print(f"\nModel Accuracy (Son Fold): {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(ytrue, yhat)
print("\nConfusion Matrix:")
print(conf_matrix)