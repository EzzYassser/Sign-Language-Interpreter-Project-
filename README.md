# Sign-Language-Interpreter-Project
A machine learning-based system to translate sign language to text/speech in real-time.

The **AI Sign Language Translator** is an innovative application designed to bridge communication gaps by translating American Sign Language (ASL) gestures into text and speech in real-time. Built using Python, this project leverages advanced technologies like computer vision, machine learning, and natural language processing to empower users with seamless interaction.

This project is ideal for developers, researchers, or enthusiasts interested in assistive technologies, offering a foundation for further enhancements in sign language translation.

## Key Features:

- **Real-Time Gesture Recognition**: Utilizes a webcam to capture hand gestures, processed through MediaPipe and a pre-trained TensorFlow model (stored as a `.pkl` file) to recognize ASL letters (A-Z).

- **Text Output**: Displays translated words in a user-friendly interface, with autocorrection powered by TextBlob to enhance accuracy.

- **Text-to-Speech (TTS)**: Converts translated text into spoken words using gTTS and Pygame, enabling auditory feedback.

- **Interactive GUI**: Developed with PyQt6, featuring a clean and intuitive interface with a home page (`HomePage.py`) and the main application (`MainGui.py`). The UI is styled with two custom stylesheets (`HomePage_Style.qss` and `MainGui_Style.qss`).

- **Smooth Prediction Processing**: Implements smoothing and redundancy filtering to ensure reliable gesture-to-text conversion, with adjustable parameters like prediction thresholds and waiting times between words.

## Technical Overview:
- **Libraries Used**: The project integrates OpenCV for video processing, MediaPipe for hand tracking, TensorFlow for machine learning, Scikit-learn for data preprocessing, TextBlob for text correction, gTTS for text-to-speech, and Pygame for audio playback.

- **Architecture**: The application is structured into three main scripts:

  - `main.py`: Entry point to launch the home page.
  
  - `HomePage.py`: Displays the welcome screen.
  
  - `MainGui.py`: Core application with camera feed, text output, and control buttons (e.g., pause, TTS, clear text).
  
- **Model**: A pre-trained TensorFlow model (`model.pkl`) is used for gesture classification, loaded dynamically using a resource path helper for compatibility with PyInstaller.

## How to Use

To get started with the AI Sign Language Translator, follow these steps:

1. **Download the Files**: Clone or download the project files from this GitHub repository.

3. **Install the Required Libraries**: Open your terminal, navigate to the project directory, and run the following command to install all necessary dependencies:

   ```bash  
   pip install PyQt6==6.8.1 opencv-python==4.7.0.72 numpy==1.23.5 tensorflow==2.12.0 mediapipe==0.10.14 scikit-learn==1.6.1 textblob==0.19.0 gTTS==2.5.4 pygame==2.6.1
  
4. **Run the Application** using PyCharm.
  
6. **Start Using It**: The application will launch with a home page. Click "Start Translation" to open the main interface. Ensure your webcam is connected, then perform ASL gestures in front of the camera. The app will translate gestures into text in real-time, display them in the text area, and allow you to use features like text-to-speech (TTS), pause, clear text, or add spaces using the on-screen buttons.


## 📦 Dependencies

| Library       | Version  | Purpose |
|--------------|----------|---------|
| [PyQt6](https://pypi.org/project/PyQt6/) | 6.8.1 | GUI Framework |
| [opencv-python](https://pypi.org/project/opencv-python/) | 4.7.0.72 | Computer Vision |
| [numpy](https://pypi.org/project/numpy/) | 1.23.5 | Numerical Operations |
| [tensorflow](https://pypi.org/project/tensorflow/) | 2.12.0 | Machine Learning |
| [mediapipe](https://pypi.org/project/mediapipe/) | 0.10.14 | Hand Tracking |
| [scikit-learn](https://pypi.org/project/scikit-learn/) | 1.6.1 | Data Processing |
| [textblob](https://pypi.org/project/textblob/) | 0.19.0 | Text Correction |
| [gTTS](https://pypi.org/project/gTTS/) | 2.5.4 | Text-to-Speech |
| [pygame](https://pypi.org/project/pygame/) | 2.6.1 | Audio Playback |
