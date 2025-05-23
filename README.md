# AI Sign Language Translator
A machine learning-based system to translate sign language to text/speech in real-time.

# AI Sign Language Translator

A machine learning-based system to translate sign language into text and speech in real-time.

The **AI Sign Language Translator** is an innovative application designed to bridge communication gaps by translating American Sign Language (ASL) gestures into text and speech in real-time. Built using Python, this project leverages advanced technologies like computer vision, machine learning, and natural language processing to empower users with seamless interaction.

## Purpose and Impact

This application aims to assist individuals who use ASL, such as those who are deaf or hard of hearing, by translating their gestures into text and spoken language in real-time. It can be particularly helpful in medical settings, enabling better communication between patients and healthcare providers. By converting ASL gestures into text and speech, the app facilitates understanding, reduces communication barriers, and promotes inclusivity.

## Key Features

- **Real-Time Gesture Recognition**: Utilizes a webcam to capture hand gestures, processed through MediaPipe and a pre-trained TensorFlow model (stored as a `.pkl` file) to recognize ASL letters (A-Z).  
- **Text Output**: Displays translated words in a user-friendly interface, with autocorrection powered by TextBlob to enhance accuracy.  
- **Text-to-Speech (TTS)**: Converts translated text into spoken words using gTTS and Pygame, enabling auditory feedback.  
- **Interactive GUI**: Developed with PyQt6, featuring a clean and intuitive interface with a home page (`HomePage.py`) and the main application (`MainGui.py`). The UI is styled with two custom stylesheets (`HomePage_Style.qss` and `MainGui_Style.qss`).  
- **Smooth Prediction Processing**: Implements smoothing and redundancy filtering to ensure reliable gesture-to-text conversion, with adjustable parameters like prediction thresholds and waiting times between words.

## Technical Overview

- **Libraries Used**: The project integrates OpenCV for video processing, MediaPipe for hand tracking, TensorFlow for machine learning, Scikit-learn for data preprocessing, TextBlob for text correction, gTTS for text-to-speech, and Pygame for audio playback.  
- **Architecture**: The application is structured into three main scripts:  
  - `main.py`: Entry point to launch the home page.  
  - `HomePage.py`: Displays the welcome screen with a gradient background.  
  - `MainGui.py`: Core application with camera feed, text output, and control buttons (e.g., pause, TTS, clear text).  
- **Model**: A pre-trained TensorFlow model (`model.pkl`) is used for gesture classification.

## Repository Structure

- **`/src`**: Contains the source code files (`main.py`, `HomePage.py`, `MainGui.py`), along with the pre-trained model (`model.pkl`) and stylesheets (`HomePage_Style.qss`, `MainGui_Style.qss`).  
- **`/exe`**: Currently not provided. This project requires running the source code directly (no pre-built executables are available).  

## Prerequisites

- **Operating System**: Windows 10/11 or macOS Ventura (13.0) or later.  
- **Python**: Version 3.11.9 (download from [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe) for Windows or [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg) for macOS).  
- **Hardware**: A webcam is required for gesture recognition. Recommended system specs:  
  - CPU: Dual-core processor or better.  
  - RAM: 4 GB minimum (8 GB recommended for smooth performance).  
  - Disk Space: At least 2 GB free for dependencies and temporary files.  
- **Additional Software**:  
  - `pip` (usually bundled with Python; ensure it’s added to your PATH).  
  - On macOS: XQuartz for PyQt6 GUI support (download from [here](https://www.xquartz.org/)).  

### Dependencies

| Library       | Version  | Purpose                  |
|---------------|----------|--------------------------|
| [PyQt6](https://pypi.org/project/PyQt6/) | 6.8.1 | GUI Framework            |
| [opencv-python](https://pypi.org/project/opencv-python/) | 4.7.0.72 | Computer Vision          |
| [numpy](https://pypi.org/project/numpy/) | 1.23.5 | Numerical Operations     |
| [tensorflow](https://pypi.org/project/tensorflow/) | 2.12.0 | Machine Learning         |
| [mediapipe](https://pypi.org/project/mediapipe/) | 0.10.14 | Hand Tracking            |
| [scikit-learn](https://pypi.org/project/scikit-learn/) | 1.6.1 | Data Processing          |
| [textblob](https://pypi.org/project/textblob/) | 0.19.0 | Text Correction          |
| [gTTS](https://pypi.org/project/gTTS/) | 2.5.4 | Text-to-Speech           |
| [pygame](https://pypi.org/project/pygame/) | 2.6.1 | Audio Playback           |
| [Python](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe) | 3.11.9 | Core Programming Language |

*Note: To avoid any kind of hassle, please use Python 3.11.9.*

## Setup and Running Instructions

### On Windows

1. **Install Python 3.11.9**:
   - Download the installer from [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe).
   - Run the installer and ensure you check "Add Python 3.11 to PATH" during installation.
   - Verify installation by opening Command Prompt and running:
     ```bash
     python --version
    It should display Python 3.11.9.

## Download the Project:

- Clone or download the repository from GitHub.
- Extract the files if downloaded as a ZIP.

## Install Dependencies:

Open Command Prompt, navigate to the project directory (e.g., `cd path\to\project`), and run:
```bash
pip install PyQt6==6.8.1 opencv-python==4.7.0.72 numpy==1.23.5 tensorflow==2.12.0 mediapipe==0.10.14 scikit-learn==1.6.1 textblob==0.19.0 gTTS==2.5.4 pygame==2.6.1

