import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QSizePolicy, QApplication, QWidget, QVBoxLayout,
                             QPushButton, QTextEdit, QLabel, QHBoxLayout,
                             QFrame, QSpacerItem)
from PyQt6.QtCore import QTimer, QSize, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QIcon
import tensorflow as tf
import mediapipe as mp
import pickle
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from textblob import TextBlob
from datetime import datetime
from gtts import gTTS
import pygame
import platform
import os
import tempfile
import threading
import warnings

# Suppress Protobuf deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

def resource_path(relative_path):
    """Get the absolute path to a resource, works for both development and PyInstaller bundled app."""
    try:
        base_path = sys._MEIPASS
        print(f"PyInstaller base path: {base_path}")
    except AttributeError:
        base_path = os.path.abspath(".")
        print(f"Development base path: {base_path}")
    full_path = os.path.join(base_path, relative_path)
    print(f"Attempting to load resource: {full_path}")
    return full_path

class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.hands = None
        self.model = None
        self.encoder = None
        self.running = True
        self.paused = False
        self.word = []
        self.last_detection_time = datetime.now()
        self.waiting_input = True
        self.prediction_threshold = 0.3
        self.wait_between_words = 3
        self.smoothing_window_size = 4
        self.autocorrection_threshold = 3
        self.categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def initialize(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error_occurred.emit("Could not open webcam")
                return False

            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75,
                max_num_hands=2
            )

            model_path = resource_path('model.pkl')
            print(f"Attempting to load model from: {model_path}")
            if not os.path.exists(model_path):
                print(f"Model file not found at: {model_path}")

            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            model_json = model_dict['model_json']
            self.model = tf.keras.models.model_from_json(model_json)
            self.model.set_weights(model_dict['model_weights'])

            categories_array = np.array(self.categories).reshape(-1, 1)
            self.encoder = OneHotEncoder(sparse_output=False)
            self.encoder.fit(categories_array)

            return True

        except Exception as e:
            self.error_occurred.emit(f"Initialization error: {str(e)}. Full context: Attempted path was {resource_path('model.pkl')}")
            return False

    def run(self):
        if not self.initialize():
            self.running = False
            return

        while self.running:
            if self.paused:
                continue
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_occurred.emit("Could not read frame")
                    continue

                frame = cv2.flip(frame, 1)
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    self.last_detection_time = datetime.now()
                    self.waiting_input = False
                    for hand_landmarks in results.multi_hand_landmarks:
                        h, w, _ = frame.shape
                        x_min, y_min, x_max, y_max = self.get_bounding_box(hand_landmarks, w, h)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                        y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                        w_lm = x_max - x_min
                        h_lm = y_max - y_min
                        if w_lm == 0 or h_lm == 0:
                            continue
                        landmarks = [((landmark.x - x_min) / w_lm, (landmark.y - y_min) / h_lm)
                                     for landmark in hand_landmarks.landmark]
                        landmarks = list(sum(landmarks, ()))
                        landmarks = np.array(landmarks).reshape(1, -1)

                        y = self.model.predict(landmarks, verbose=0)
                        if max(y[0]) > self.prediction_threshold:
                            y_decoded = self.encoder.inverse_transform(y)[0][0]
                            self.word.append(y_decoded)
                            cv2.putText(frame, y_decoded, (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.putText(frame, f"P: {max(y[0]):.2f}", (300, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                        (0, 255, 0), 2)
                else:
                    if (datetime.now() - self.last_detection_time).total_seconds() > self.wait_between_words and not self.waiting_input:
                        processed_word = self.process_predicted_word(self.word, self.smoothing_window_size)
                        if processed_word:
                            self.prediction_ready.emit(processed_word)
                        self.word = []
                        self.waiting_input = True

                self.frame_ready.emit(frame)

            except Exception as e:
                self.error_occurred.emit(f"Processing error: {str(e)}")

    def get_bounding_box(self, landmarks, image_width, image_height, scale=1.0):
        x_coords = [landmark.x * image_width for landmark in landmarks.landmark]
        y_coords = [landmark.y * image_height for landmark in landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        box_size = max(x_max - x_min, y_max - y_min) * scale
        half_size = int(box_size // 2)
        x_min_new = max(x_center - half_size, 0)
        x_max_new = min(x_center + half_size, image_width)
        y_min_new = max(y_center - half_size, 0)
        y_max_new = min(y_center + half_size, image_height)
        return x_min_new, y_min_new, x_max_new, y_max_new

    def smooth_predictions(self, predictions, window_size=4):
        smoothed = []
        for i in range(len(predictions) - window_size + 1):
            window = predictions[i:i + window_size]
            most_common = Counter(window).most_common(1)[0][0]
            smoothed.append(most_common)
        return smoothed

    def remove_redundant(self, predictions, threshold=3):
        filtered = []
        last_char = predictions[0]
        count = 0
        for char in predictions:
            if char == last_char:
                count += 1
            else:
                if count >= threshold:
                    filtered.append(last_char)
                count = 1
                last_char = char
        if count >= threshold:
            filtered.append(last_char)
        return filtered

    def process_predicted_word(self, letters_list, window_size=4):
        if len(letters_list) < window_size:
            return ''.join(letters_list).lower()
        else:
            smoothing = self.smooth_predictions(letters_list, window_size)
            filter_redundants = self.remove_redundant(smoothing)
            if len(filter_redundants) <= self.autocorrection_threshold:
                return ''.join(filter_redundants).lower()
            autocorrected = str(TextBlob(''.join(filter_redundants).lower()).correct())
            return autocorrected

    def pause_resume(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        self.wait()

def text_to_speech(word, lang='en'):
    if not word:
        print("No text provided for TTS.")
        return
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
    try:
        tts = gTTS(text=word, lang=lang)
        tts.save(temp_file)
        if not os.path.exists(temp_file):
            print(f"Error: Temporary file {temp_file} was not created.")
            return
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"TTS Playback Error: {str(e)}")
    finally:
        pygame.mixer.quit()
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            else:
                print(f"Warning: Temporary file {temp_file} does not exist during cleanup.")
        except Exception as e:
            print(f"Error removing temp file: {str(e)}")

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        print("Initializing SignLanguageApp...")
        self.initUI()

        self.produced_words = []
        self.is_speaking = False

        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.prediction_ready.connect(self.add_prediction)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def initUI(self):
        self.setWindowTitle("AI Sign Language Translator")
        self.showMaximized()

        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(10)
        left_panel_frame = QFrame()
        left_panel_frame.setFixedWidth(100)
        left_panel_frame.setLayout(left_panel)

        self.home_button = QPushButton()
        self.home_button.setObjectName("LeftButton")
        try:
            self.home_button.setIcon(QIcon(resource_path("Icons/Home2.png")))
        except:
            print("Warning: Home2.png not found.")
        self.home_button.setIconSize(QSize(30, 30))
        self.home_button.setFixedSize(45, 45)

        self.tts_button = QPushButton()
        self.tts_button.setObjectName("LeftButton")
        try:
            self.tts_button.setIcon(QIcon(resource_path("Icons/TTS.png")))
        except:
            print("Warning: TTS.png not found.")
        self.tts_button.setIconSize(QSize(35, 35))
        self.tts_button.setFixedSize(45, 45)
        self.tts_button.clicked.connect(self.play_tts)

        self.pause_button = QPushButton()
        self.pause_button.setObjectName("LeftButton")
        try:
            self.pause_button.setIcon(QIcon(resource_path("Icons/Pause.png")))
        except:
            print("Warning: Pause.png not found.")
        self.pause_button.setIconSize(QSize(30, 30))
        self.pause_button.setFixedSize(45, 45)
        self.pause_button.clicked.connect(self.toggle_pause)

        left_panel.addWidget(self.home_button)
        left_panel.addWidget(self.tts_button)
        left_panel.addWidget(self.pause_button)
        left_panel.addStretch()

        main_layout.addWidget(left_panel_frame)

        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.Shape.VLine)
        vertical_line.setFrameShadow(QFrame.Shadow.Sunken)
        vertical_line.setLineWidth(1)
        vertical_line.setStyleSheet("background-color: #5C6979;")
        main_layout.addWidget(vertical_line)

        center_layout = QVBoxLayout()

        self.video_label = QLabel("Camera Feed Placeholder")
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(True)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_output = QTextEdit()
        self.text_output.setFixedHeight(200)
        self.text_output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.text_output.setStyleSheet("font-size: 18px; background-color: #222; color: white;")
        self.text_output.setFocus()  # Set initial focus for cursor visibility
        self.text_output.keyPressEvent = self.block_key_input  # Block manual key input

        center_layout.addWidget(self.video_label, 4)
        center_layout.addWidget(self.text_output, 1)

        right_panel_layout = QVBoxLayout()
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(10)
        right_panel_frame = QFrame()
        right_panel_frame.setFixedWidth(75)
        right_panel_frame.setLayout(right_panel_layout)

        right_panel_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.clear_button = QPushButton()
        self.clear_button.setObjectName("RightButton")
        try:
            self.clear_button.setIcon(QIcon(resource_path("Icons/ClearText.png")))
        except:
            print("Warning: ClearText.png not found.")
        self.clear_button.setIconSize(QSize(30, 30))
        self.clear_button.setFixedSize(45, 45)
        self.clear_button.clicked.connect(self.clear_text)

        self.space_button = QPushButton()
        self.space_button.setObjectName("RightButton")
        try:
            self.space_button.setIcon(QIcon(resource_path("Icons/Space.png")))
        except:
            print("Warning: Space.png not found.")
        self.space_button.setIconSize(QSize(35, 20))
        self.space_button.setFixedSize(45, 45)
        self.space_button.clicked.connect(self.insert_space)

        self.backspace_button = QPushButton()
        self.backspace_button.setObjectName("RightButton")
        try:
            self.backspace_button.setIcon(QIcon(resource_path("Icons/Backspace.png")))
        except:
            print("Warning: Backspace.png not found.")
        self.backspace_button.setIconSize(QSize(30, 30))
        self.backspace_button.setFixedSize(45, 45)
        self.backspace_button.clicked.connect(self.backspace_text)

        right_panel_layout.addWidget(self.clear_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        right_panel_layout.addWidget(self.space_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        right_panel_layout.addWidget(self.backspace_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        right_panel_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        main_layout.addLayout(center_layout)
        main_layout.addWidget(right_panel_frame)
        self.setLayout(main_layout)

        print("Loading stylesheet...")
        try:
            with open(resource_path("MainGui_Style.qss"), "r") as f:
                self.setStyleSheet(f.read())
            print("Stylesheet loaded.")
        except FileNotFoundError:
            print("Warning: MainGui_Style.qss not found. Using default styles.")

    def block_key_input(self, event):
        """Prevent manual key input in QTextEdit to mimic read-only behavior."""
        event.ignore()  # Ignore all key events to prevent editing

    def toggle_pause(self):
        self.worker.pause_resume()
        try:
            if self.worker.paused:
                self.pause_button.setIcon(QIcon(resource_path("Icons/Resume.png")))
            else:
                self.pause_button.setIcon(QIcon(resource_path("Icons/Pause.png")))
        except:
            print("Warning: Pause/Resume icon not found.")

    def play_tts(self):
        if self.is_speaking:
            print("TTS is already playing, please wait.")
            return

        text = self.text_output.toPlainText().strip()
        if not text:
            print("No text to speak.")
            return

        self.is_speaking = True
        try:
            if platform.system() == "Emscripten":
                text_to_speech(text)
            else:
                threading.Thread(target=text_to_speech, args=(text,), daemon=True).start()
        except Exception as e:
            print(f"TTS Error: {str(e)}")
        finally:
            self.is_speaking = False

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    @pyqtSlot(str)
    def add_prediction(self, word):
        if self.produced_words and self.produced_words[-1] != "":
            self.produced_words[-1] += word
            self.produced_words.append("")  # Add space after appending to existing word
        else:
            self.produced_words.append(word)
            self.produced_words.append("")  # Add space after new word
        self.update_text_output()

    @pyqtSlot(str)
    def handle_error(self, error_msg):
        print(f"Error: {error_msg}")
        error_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Wrap text to avoid overflow
        words = error_msg.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + word) < 50:  # Adjust based on screen width
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        y = 200  # Starting y-coordinate
        for line in lines:
            cv2.putText(error_image, line, (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30  # Move to next line
        h, w, ch = error_image.shape
        q_img = QImage(error_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_text_output(self):
        if self.produced_words and self.produced_words[0]:
            words = self.produced_words.copy()
            words[0] = words[0].capitalize()
            self.text_output.setPlainText(' '.join(words))
            cursor = self.text_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.text_output.setTextCursor(cursor)
            self.text_output.ensureCursorVisible()
            self.text_output.setFocus()  # Ensure focus to keep cursor visible
        else:
            self.text_output.setPlainText(' '.join(self.produced_words))
            cursor = self.text_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.text_output.setTextCursor(cursor)
            self.text_output.ensureCursorVisible()
            self.text_output.setFocus()  # Ensure focus to keep cursor visible

    def clear_text(self):
        self.produced_words = []
        self.text_output.setPlainText("")
        cursor = self.text_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.text_output.setTextCursor(cursor)
        self.text_output.ensureCursorVisible()
        self.text_output.setFocus()  # Ensure focus to keep cursor visible

    def insert_space(self):
        if self.produced_words:
            self.produced_words.append("")
            self.update_text_output()

    def backspace_text(self):
        if self.produced_words:
            if self.produced_words[-1] == "":
                self.produced_words.pop()
            elif len(self.produced_words[-1]) > 0:
                self.produced_words[-1] = self.produced_words[-1][:-1]
                if self.produced_words[-1] == "":
                    self.produced_words.pop()
            self.update_text_output()

    def closeEvent(self, event):
        print("Closing application...")
        self.worker.stop()
        pygame.mixer.quit()
        event.accept()

if __name__ == '__main__':
    print("Starting application...")
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())