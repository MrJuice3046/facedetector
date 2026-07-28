# Emotion HUD App 🤖

A real-time, sci-fi inspired AI emotion detection heads-up display (HUD) that runs directly on your webcam! 

## Features
- **Live Facial Tracking**: Smoothly tracks up to 5 faces simultaneously using MediaPipe.
- **Deep AI Emotion Analysis**: Uses DeepFace to classify 13 distinct core and complex emotions (Happy, Sad, Angry, Excited, Frustrated, Panicked, Bored, etc.).
- **Temporal Smoothing**: Applies a rolling-average sliding window to stabilize AI predictions and eliminate jitter.
- **Auto-Alignment**: Surgically crops and aligns faces horizontally to ensure extremely high emotion detection accuracy.
- **Ayanakoji Mode**: Put your hand over your face to instantly trigger the emotionless "Ayanakoji" override mode!

## Installation

1. Make sure you have Python installed on your computer.
2. Clone or download this repository.
3. Open your terminal/command prompt inside the folder where you downloaded these files.
4. Install the required AI libraries by running:

```bash
pip install -r requirements.txt
```

## How to Run

Once the requirements are installed, simply run the Python script from your terminal:

```bash
python hud_app.py
```

*Note: The very first time you run this, it will take about 20-30 seconds to download the AI model weights from Google and compile the neural network into your memory. Every run after that will boot instantly!*

## Controls
- Press **`q`** to quit the application.
