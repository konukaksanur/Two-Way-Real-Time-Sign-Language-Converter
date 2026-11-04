import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, Input
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import (
    LSTM, 
    Dense, 
    Dropout,
    LayerNormalization,
    MultiHeadAttention, 
    GlobalAveragePooling1D, 
    Add, 
    Concatenate
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Define data path and load actions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "keypoints_175_40")

actions = np.array(sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
]))
label_map = {label: num for num, label in enumerate(actions)}

# Data processing
sequences, labels = [], []
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

        if len(window) > 0:  # Add sequence if it has frames
            sequences.append(window)
            labels.append(label_map[action])


# Validate the shape of training sequences
if not sequences:
    raise ValueError("No training sequences were found.")

if not all(np.asarray(seq).shape == (40, 307) for seq in sequences):
    raise ValueError("All sequences must contain 40 frames and 307 features.")

# Convert training sequences to a NumPy array
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
print(f"X_left.shape: {X_left.shape}")

#######################################################################################

X_p_keypoint = X[:, :, 0:99]
X_p_angle = X[:, :, 225:229]
X_p_distance = X[:, :, 299:307]
X_p_intra_hand = X[:, :, 293:299]

X_pose = np.concatenate((X_p_keypoint, X_p_angle, X_p_distance, X_p_intra_hand), axis=2)
print(f"X_pose.shape: {X_pose.shape}")
#######################################################################################

num_samples, timesteps, num_features = X.shape
print(f"num_samples: {num_samples}")
print(f"timesteps: {timesteps}")
print(f"num_features: {num_features}")

y = to_categorical(labels).astype(int)


# Step 1: Reserve 10% of the dataset for independent testing
(
    X_right_dev, X_right_test,
    X_left_dev, X_left_test,
    X_pose_dev, X_pose_test,
    y_dev, y_test,
    labels_dev, labels_test
) = train_test_split(
    X_right, X_left, X_pose, y, np.array(labels),
    test_size=0.10,
    stratify=labels,
    random_state=42
)

# Step 2: Split the remaining 90% into training and validation
# Approximately 11.11% of the remaining 90% is used for validation
(
    X_right_train, X_right_val,
    X_left_train, X_left_val,
    X_pose_train, X_pose_val,
    y_train, y_val
) = train_test_split(
    X_right_dev, X_left_dev, X_pose_dev, y_dev,
    test_size=0.111111,
    stratify=labels_dev,
    random_state=42
)

# Verify class distribution across datasets
train_classes = np.argmax(y_train, axis=1)
val_classes = np.argmax(y_val, axis=1)
test_classes = np.argmax(y_test, axis=1)

print(f"Training classes: {len(np.unique(train_classes))}")
print(f"Validation classes: {len(np.unique(val_classes))}")
print(f"Test classes: {len(np.unique(test_classes))}")

# Display dataset sizes
print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples: {len(y_test)}")


# --- Inputs ---
input_right = Input(shape=(X_right.shape[1], X_right.shape[2]))
input_left = Input(shape=(X_left.shape[1], X_left.shape[2]))
input_pose = Input(shape=(X_pose.shape[1], X_pose.shape[2]))


# --- Right-hand LSTM branch ---
x_r = LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01))(input_right)
x_r = LayerNormalization(axis=1)(x_r)
x_r = Dropout(0.4)(x_r)

# --- Left-hand LSTM branch ---
x_l = LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01))(input_left)
x_l = LayerNormalization(axis=1)(x_l)
x_l = Dropout(0.4)(x_l)

# --- Pose LSTM branch ---
x_p = LSTM(128, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.01))(input_pose)
x_p = LayerNormalization(axis=1)(x_p)
x_p = Dropout(0.4)(x_p)

# --- Combine features using attention ---
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

# --- Training ---
history = model.fit(
    [X_right_train, X_left_train, X_pose_train], y_train,
    validation_data=([X_right_val, X_left_val, X_pose_val], y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],    
    verbose=1
)

# Evaluate the final model on the independent test set
test_loss, test_acc = model.evaluate(
    [X_right_test, X_left_test, X_pose_test],
    y_test,
    verbose=0
)

print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save the trained model
MODEL_PATH = os.path.join(BASE_DIR, "action4.h5")
model.save(MODEL_PATH)

# Evaluate the model
yhat = model.predict([X_right_test, X_left_test, X_pose_test])
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

conf_matrix = confusion_matrix(ytrue, yhat)
print("Confusion Matrix:")
print(conf_matrix)


plt.figure(figsize=(8, 4))
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_over_epochs.png', dpi=300)
plt.show()

# --- Loss Graph ---
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_over_epochs.png', dpi=300)
plt.show()