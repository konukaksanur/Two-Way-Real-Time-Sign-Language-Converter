import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2

from scipy.interpolate import CubicSpline

# Define data path and load actions
DATA_PATH = "keypoints"
actions = np.array(os.listdir(DATA_PATH))  # Get class names from subdirectories
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
X = (X - X.min()) / (X.max() - X.min())  # Normalize data
y = to_categorical(labels).astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the model
model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(40, X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, callbacks=[tb_callback])

# Save the model
model.save('actions.h5')

# Evaluate the model
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# Print accuracy and confusion matrix
accuracy = accuracy_score(ytrue, yhat)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(ytrue, yhat)
print("Confusion Matrix:")
print(conf_matrix)