# AI Sign Language Translator

A machine learning-based system to translate sign language into text and speech in real-time.

The **AI Sign Language Translator** is an innovative application designed to bridge communication gaps by translating American Sign Language (ASL) gestures into text and speech in real-time. Built using Python, this project leverages advanced technologies like computer vision, machine learning, and natural language processing to empower users with seamless interaction.
<br><br><br>
## Purpose and Impact
<br>
This application aims to assist individuals who use ASL, such as those who are deaf or hard of hearing, by translating their gestures into text and spoken language in real-time. It can be particularly helpful in medical settings, enabling better communication between patients and healthcare providers. By converting ASL gestures into text and speech, the app facilitates understanding, reduces communication barriers, and promotes inclusivity.
<br><br><br>
## Key Features

- **Real-Time Gesture Recognition**: Utilizes a webcam to capture hand gestures, processed through MediaPipe and a pre-trained TensorFlow model (stored as a `.pkl` file) to recognize ASL letters (A-Z).  
- **Text Output**: Displays translated words in a user-friendly interface, with autocorrection powered by TextBlob to enhance accuracy.  
- **Text-to-Speech (TTS)**: Converts translated text into spoken words using gTTS and Pygame, enabling auditory feedback.  
- **Interactive GUI**: Developed with PyQt6, featuring a clean and intuitive interface with a home page (`HomePage.py`) and the main application (`MainGui.py`). The UI is styled with two custom stylesheets (`HomePage_Style.qss` and `MainGui_Style.qss`).  
- **Smooth Prediction Processing**: Implements smoothing and redundancy filtering to ensure reliable gesture-to-text conversion, with adjustable parameters like prediction thresholds and waiting times between words.
<br><br><br><br>
## Technical Overview

- **Libraries Used**: The project integrates OpenCV for video processing, MediaPipe for hand tracking, TensorFlow for machine learning, Scikit-learn for data preprocessing, TextBlob for text correction, gTTS for text-to-speech, and Pygame for audio playback.  
- **Architecture**: The application is structured into three main scripts:  
  - `main.py`: Entry point to launch the home page.  
  - `HomePage.py`: Displays the welcome screen with a gradient background.  
  - `MainGui.py`: Core application with camera feed, text output, and control buttons (e.g., pause, TTS, clear text).  
- **Model**: A pre-trained TensorFlow model (`model.pkl`) is used for gesture classification, loaded dynamically using a resource path helper for compatibility with PyInstaller.
<br><br><br>
## Repository Structure

- **`/src`**: Contains the source code files (`main.py`, `HomePage.py`, `MainGui.py`), along with the pre-trained model (`model.pkl`) and stylesheets (`HomePage_Style.qss`, `MainGui_Style.qss`).  
- **`/exe`**: Currently not provided. This project requires running the source code directly (no pre-built executables are available).
<br><br><br><br>
## Prerequisites

- **Operating System**: Windows 10/11 or macOS Ventura (13.0) or later.  
- **Python**: Version 3.11.9 (download from [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe) for Windows or [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg) for macOS).  
- **Hardware**: A webcam is required for gesture recognition. Recommended system specs:  
  - CPU: Dual-core processor or better.  
  - RAM: 4 GB minimum (8 GB recommended for smooth performance).  
  - Disk Space: At least 2 GB free for dependencies and temporary files.  
- **Additional Software**:  
  - `pip` (usually bundled with Python; ensure it's added to your PATH).  
  - On macOS: XQuartz for PyQt6 GUI support (download from [here](https://www.xquartz.org/)).  
<br><br><br><br>
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
<br><br><br><br>
## 🧠 Model Development

### 📊 Training Notebook
Explore the model training process:  
[`training.ipynb`](./notebooks/training.ipynb)

### 📂 Datasets
Download the ASL datasets used:  
[`asl_dataset/`](./datasets/ASL_Datasets/)  

<br><br><br><br>
## Setup and Running Instructions

### On Windows

1. **Install Python 3.11.9**:
   - Download the installer from [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe).
   - Run the installer and ensure you check "Add Python 3.11 to PATH" during installation.
   - Verify installation by opening Command Prompt and running:
     ```bash
     python --version
     ```
     It should display `Python 3.11.9`.

2. **Download the Project**:
   - Clone or download the repository from GitHub.
   - Extract the files if downloaded as a ZIP.

3. **Install Dependencies**:
   - Open Command Prompt, navigate to the project directory (e.g., `cd path\to\project`), and run:
     ```bash
     pip install PyQt6==6.8.1 opencv-python==4.7.0.72 numpy==1.23.5 tensorflow==2.12.0 mediapipe==0.10.14 scikit-learn==1.6.1 textblob==0.19.0 gTTS==2.5.4 pygame==2.6.1
     ```

4. **Configure Webcam**:
   - Ensure your webcam is connected and drivers are installed (Windows usually handles this automatically).
   - Test your webcam using an app like the Camera app to confirm it's working.

5. **Run the Application**:
   - In Command Prompt, navigate to the `/src` directory (e.g., `cd path\to\project\src`), and run:
     ```bash
     python main.py
     ```
   - Alternatively, open the project in PyCharm, navigate to `/src`, and run `main.py` from there.
<br><br>
### On macOS

1. **Install Python 3.11.9**:
   - Download the installer from [here](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg).
   - Run the installer and follow the prompts.
   - Verify installation by opening Terminal and running:
     ```bash
     python3 --version
     ```
     It should display `Python 3.11.9`.

2. **Install XQuartz (for PyQt6 GUI)**:
   - Download and install XQuartz from [here](https://www.xquartz.org/).
   - After installation, log out and log back into your macOS session to apply changes.

3. **Download the Project**:
   - Clone or download the repository from GitHub.
   - Extract the files if downloaded as a ZIP.

4. **Install Dependencies**:
   - Open Terminal, navigate to the project directory (e.g., `cd path/to/project`), and run:
     ```bash
     pip3 install PyQt6==6.8.1 opencv-python==4.7.0.72 numpy==1.23.5 tensorflow==2.12.0 mediapipe==0.10.14 scikit-learn==1.6.1 textblob==0.19.0 gTTS==2.5.4 pygame==2.6.1
     ```

5. **Configure Webcam**:
   - Ensure your webcam is connected.
   - Grant camera permissions: Go to `System Settings > Privacy & Security > Camera` and ensure your terminal or PyCharm has access.

6. **Run the Application**:
   - In Terminal, navigate to the `/src` directory (e.g., `cd path/to/project/src`), and run:
     ```bash
     python3 main.py
     ```
   - Alternatively, open the project in PyCharm, navigate to `/src`, and run `main.py` from there.

<br><br><br><br>
## Using the Application

- The application will launch with a home page.
- Click "Start Translation" to open the main interface.
- Ensure your webcam is connected, then perform ASL gestures in front of the camera.
- The app will translate gestures into text in real-time, display them in the text area, and allow you to use features like text-to-speech (TTS), pause, clear text, or add spaces using the on-screen buttons.

<br><br><br><br>
## Troubleshooting Tips

- **Webcam Not Detected**:
  - Ensure your webcam is connected and drivers are installed.
  - On macOS, verify camera permissions in `System Settings > Privacy & Security > Camera`.
  - Test your webcam with another app to confirm it's working.
- **Dependency Installation Fails**:
  - Ensure `pip` is installed and up-to-date: `python -m ensurepip --upgrade` or `python3 -m ensurepip --upgrade`.
  - Check your internet connection.
  - Try installing one dependency at a time to identify the problematic package.
- **PyQt6 GUI Issues on macOS**:
  - Ensure XQuartz is installed and you've logged out and back in after installation.
- **Application Runs Slowly**:
  - Close other resource-intensive applications.
  - Ensure your hardware meets the recommended specs (see Prerequisites).
- **Model or Stylesheet Not Found**:
  - Ensure `model.pkl`, `HomePage_Style.qss`, and `MainGui_Style.qss` are in the `/src` directory alongside the Python scripts.

<br><br><br><br>
## Demo / Usage Video

To better understand how the application works, please check the video below showing the app in action:  
- [Demo Video Link](https://www.youtube.com/your-video-link)

<br><br><br><br>
## Limitations

- **Gesture Recognition**: The app currently supports only ASL letters (A-Z) and may not accurately recognize full words or complex gestures.
- **Hardware Requirements**: A webcam is required, and the app may run slower on older hardware due to real-time processing.
- **Accuracy**: While autocorrection improves output, some gestures may still be misrecognized, especially with fast movements or overlapping hands.

<br><br><br><br>
## Future Improvements

- Expand gesture recognition to include full ASL words, numbers, and common phrases.  
- Fix the non-functional home button to enable navigation back to the home page.  
- Optimize performance for lower-end devices.  

<br><br><br><br>
## Contact/Support

If you encounter any issues or have suggestions for improvement, please feel free to reach out:  
- **Email**: [ezzeldin.yasser@hotmail.com] or [dramirasaad103@gmail.com]  
- **GitHub Issues**: Feel free to open an issue on this repository for technical support or feedback.
