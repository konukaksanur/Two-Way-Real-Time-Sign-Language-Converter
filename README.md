
<h1 align="center">Two-Way Real-Time Sign Language Converter</h1>

<p align="justify">
This graduation project, developed during the final year of a Bachelor's degree in Computer Engineering, aims to make communication between Turkish Sign Language (TSL) users and non-signers more accessible. The system combines real-time sign recognition with 3D animations that visualize Turkish Sign Language gestures.

The recognition model covers 175 Turkish Sign Language gestures and achieved 93.37% test accuracy.

## Demo

The system supports two directions of conversion:

- **Sign to Text:** Recognizes signs through a webcam and displays the corresponding Turkish words in real time.


https://github.com/user-attachments/assets/a4e82813-4792-40e6-8e98-96b12bdca093


- **Text to Sign:** Plays 3D sign animations for supported Turkish words entered by the user.


https://github.com/user-attachments/assets/978531a6-425e-4515-a229-d03a771d8479



## Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/konukaksanur/Two-Way-Real-Time-Sign-Language-Converter.git
cd Two-Way-Real-Time-Sign-Language-Converter
```

**2. Install dependencies**

Creating a virtual environment is recommended but optional.

```bash
# Optional: Create and activate a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

**3. Run the application**

From the repository root:

```bash
python -m web_app.app
```

Open `http://127.0.0.1:5001` in your browser.

## Dataset

The project uses a subset of the **BosphorusSign22k** dataset (Özdemir et al., 2020), consisting of **175 Turkish Sign Language gesture classes**. The original dataset contains 22,542 video samples representing 744 signs performed by six signers.

The selected videos were processed to extract structured motion features for model training.

The original videos and processed keypoint sequences are not distributed in this repository due to privacy and data-sharing considerations.



## Data Preprocessing

The preprocessing pipeline uses **OpenCV**, **MediaPipe Holistic**, and **NumPy** to transform sign language videos into fixed-length feature sequences.

Each video is processed through the following steps:

1. **Frame Sampling:** 40 frames are sampled from each video.
2. **Landmark Extraction:** MediaPipe Holistic extracts 33 pose landmarks and 21 landmarks per hand, resulting in 75 landmarks represented by 3D coordinates.
3. **Feature Engineering:** To help the model learn gesture patterns more effectively, 82 additional features are calculated to capture geometric relationships between hand and body landmarks. These include finger and arm joint angles, distances between fingertips, relative hand positions, and distances between the hands and body.
4. **Data Storage:** The extracted features are stored as NumPy (`.npy`) arrays for model training.

By combining **225 landmark coordinates** with **82 engineered features**, each frame is represented by a **307-dimensional feature vector**, resulting in a sequence of **40 × 307 features** per gesture.


## Model Architecture & Model Training

A multi-input deep learning model was developed using **TensorFlow** and **Keras**.

Instead of processing all features through a single network, the input was divided into three groups:

- Right-hand features
- Left-hand features
- Body pose and related geometric features

Each input group is processed by a separate **LSTM branch** to capture temporal movement patterns.

The resulting representations are combined using **Multi-Head Attention**, followed by pooling and fully connected layers for classification.

<img width="2172" height="724" alt="a_clean_flat_vector_diagram_on_a_white_background" src="https://github.com/user-attachments/assets/a12f9560-1c3c-47cf-9ef6-2ba60f136021" />


### Training Configuration

| Parameter | Value |
|---|---|
| Gesture classes | 175 |
| Frames per sample | 40 |
| Features per frame | 307 |
| Train / Validation / Test split | 80% / 10% / 10% |
| Optimizer | Adam |
| Initial learning rate | 0.0005 |
| Batch size | 64 |
| Maximum epochs | 100 |

The training process uses **EarlyStopping** and **ReduceLROnPlateau** based on validation loss.

Implementation: `sign_language_recognition/train.py`.

## Results

The final model achieved **93.37% test accuracy** across 175 Turkish Sign Language gesture classes.

<img width="874" height="504" alt="Screenshot 2026-11-05 at 03 37 20" src="https://github.com/user-attachments/assets/ae14641a-357c-445b-bd23-57f43a5dd3cf" />


For real-time recognition, the application processes a rolling window of **40 frames**. Predictions are accepted only when the confidence score exceeds **0.99**, with a **20-frame lock** to reduce repeated predictions.

Recognition performance may vary depending on lighting conditions, camera positioning, background complexity, and individual signing styles.

## Web Interface

A lightweight web interface was developed using **Flask, HTML, CSS, and JavaScript** to demonstrate the system's bidirectional conversion capabilities. It integrates webcam-based gesture recognition and sequential playback of sign language animations created using **Unity 3D**.

<!-- Add application screenshots here. -->

## Reference

Özdemir, O., Kındıroğlu, A. A., Camgöz, N. C., & Akarun, L. (2020). *BosphorusSign22k Sign Language Recognition Dataset*. Proceedings of the 9th Workshop on the Representation and Processing of Sign Languages, 181–188.
</p>
