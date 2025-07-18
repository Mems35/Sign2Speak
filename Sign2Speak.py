import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
import threading # Though not strictly used for ML inference threading yet, it's good to keep for future expansion

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global flag to control loop
running = True

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign2Speak - ASL Translator")

        # --- UI Improvements (from critique) ---
        self.video_label = Label(root, bg="black") # Add background for better visual
        self.video_label.pack(pady=10) # Add some padding

        self.translation_label = Label(root, text="Predicted Gesture: (No hand detected)",
                                       font=("Arial", 24, "bold"), fg="blue") # Larger, bold, colored text
        self.translation_label.pack(pady=20)
        # --- End UI Improvements ---

        # --- Robust Camera Initialization (from critique) ---
        self.capture = cv2.VideoCapture(0) # Try default camera
        if not self.capture.isOpened():
            print("Error: Could not open video device. Trying another index...")
            self.capture = cv2.VideoCapture(1) # Try second camera
            if not self.capture.isOpened():
                # Display an error message box using tkinter
                messagebox.showerror("Camera Error", "Could not open any camera. Please ensure a webcam is connected and not in use by another application.")
                self.root.destroy()
                return # Exit init if camera fails
        # --- End Robust Camera Initialization ---

        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        # Start video loop only if camera was successfully opened
        self.video_loop()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def video_loop(self):
        global running
        if running:
            ret, frame = self.capture.read()
            if ret:
                # Flip for selfie-view and convert to RGB
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = None
                # --- Error Handling for MediaPipe Processing (from critique) ---
                try:
                    results = self.hands.process(rgb)
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")
                    # You might want to display a temporary message on screen
                    # or log this error more persistently.
                # --- End Error Handling ---

                # Draw landmarks and update translation
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks,
                                                  mp_hands.HAND_CONNECTIONS)

                    # Placeholder gesture prediction logic
                    # gesture = self.classify_gesture(hand_landmarks)
                    gesture = "A" # mock output for now
                    self.translation_label.config(text=f"Predicted Gesture: {gesture}")
                else:
                    self.translation_label.config(text="Predicted Gesture: (No hand detected)") # Update when no hand

                # Convert to ImageTk for Tkinter display
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Repeat the loop after 10ms
            self.root.after(10, self.video_loop)

    def on_closing(self):
        global running
        running = False
        if self.capture.isOpened(): # Ensure capture is open before releasing
            self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

