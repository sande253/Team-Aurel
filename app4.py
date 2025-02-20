import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from collections import deque

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

st.title("AR T-Shirt Try-On with Pattern Blending")

# Function to blend pattern onto the shirt
def blend_pattern_on_shirt(shirt_image, pattern_image):
    shirt = np.array(shirt_image.convert("RGBA"))
    pattern = np.array(pattern_image.convert("RGBA"))

    pattern = cv2.resize(pattern, (shirt.shape[1], shirt.shape[0]))

    if shirt.shape[2] == 4:
        shirt_alpha = shirt[:, :, 3]
    else:
        shirt_alpha = 255 * np.ones((shirt.shape[0], shirt.shape[1]), np.uint8)

    _, mask = cv2.threshold(shirt_alpha, 230, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_inv = cv2.bitwise_not(mask)
    shirt_region = cv2.bitwise_and(shirt, shirt, mask=mask)
    pattern_region = cv2.bitwise_and(pattern, pattern, mask=mask_inv)
    blended = cv2.add(shirt_region, pattern_region)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGRA2RGBA)

    return Image.fromarray(blended_rgb)

# File Uploaders
shirt_file = st.file_uploader("Upload Shirt Image", type=["png", "jpg"])
pattern_file = st.file_uploader("Upload Pattern Image", type=["png", "jpg"])

if shirt_file and pattern_file:
    shirt_image = Image.open(shirt_file).convert("RGBA")
    pattern_image = Image.open(pattern_file).convert("RGBA")
    blended_shirt = blend_pattern_on_shirt(shirt_image, pattern_image)

    st.image(blended_shirt, caption="✨ Blended Shirt", use_container_width=True)

# Start webcam
start_webcam = st.button("Start Webcam")

if start_webcam and shirt_file and pattern_file:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera settings.")
    else:
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Webcam")
        
        # Convert blended_shirt to numpy array once outside the loop
        blended_shirt_np = np.array(blended_shirt)
        
        # Simple stabilization variables
        width_history = deque(maxlen=5)
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.7 and \
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.7:
                    
                    h, w, _ = frame.shape
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]

                    left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
                    right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
                    nose_x = int(nose.x * w)

                    # Calculate center point between shoulders
                    center_x = (left_x + right_x) // 2
                    center_y = (left_y + right_y) // 2
                    
                    # Get current shoulder width
                    current_width = abs(right_x - left_x)
                    width_history.append(current_width)
                    
                    # Calculate average width for stability
                    avg_width = sum(width_history) / len(width_history)
                    
                    # Detect rotation based on nose position relative to shoulder center
                    # Calculate as percentage offset from center (0 = centered, 1 = far right, -1 = far left)
                    rotation_offset = (nose_x - center_x) / (current_width * 0.5) if current_width > 0 else 0
                    
                    # Adjust width multiplier based on rotation - REDUCED TO FIX OVERSIZING
                    # When facing camera: use normal multiplier (1.6)
                    # When turned sideways: use slightly larger multiplier to compensate for narrower shoulders
                    base_multiplier = 1.6  # Base size multiplier (REDUCED from 1.8)
                    rotation_compensation = abs(rotation_offset) * 0.2  # Subtle compensation factor
                    width_multiplier = base_multiplier + rotation_compensation
                    
                    # Cap the multiplier to prevent excessive sizing
                    width_multiplier = min(width_multiplier, 1.8)
                    
                    # Calculate shirt dimensions
                    shirt_width = int(avg_width * width_multiplier)
                    shirt_height = int(shirt_width * (blended_shirt_np.shape[0] / blended_shirt_np.shape[1]))

                    # Position the shirt
                    shirt_x = center_x - (shirt_width // 2)
                    shirt_y = center_y - (shirt_height // 4)

                    # Boundary checks
                    if shirt_x < 0: shirt_x = 0
                    if shirt_y < 0: shirt_y = 0
                    if shirt_x + shirt_width > w: shirt_width = w - shirt_x
                    if shirt_y + shirt_height > h: shirt_height = h - shirt_y
                    
                    # Only proceed if we have valid dimensions
                    if shirt_width > 0 and shirt_height > 0:
                        try:
                            # Resize shirt
                            resized_shirt = cv2.resize(blended_shirt_np, (shirt_width, shirt_height))
                            
                            # Alpha blending for AR overlay
                            if resized_shirt.shape[2] == 4:  # With alpha channel
                                alpha = resized_shirt[:, :, 3] / 255.0
                                alpha = np.stack([alpha, alpha, alpha], axis=-1)
                                
                                roi = frame[shirt_y:shirt_y+shirt_height, shirt_x:shirt_x+shirt_width]
                                
                                # Make sure ROI dimensions match resized_shirt
                                if roi.shape[0] == resized_shirt.shape[0] and roi.shape[1] == resized_shirt.shape[1]:
                                    # Blend using alpha: foreground*alpha + background*(1-alpha)
                                    foreground = resized_shirt[:, :, :3]
                                    blended = (foreground * alpha + roi * (1 - alpha)).astype(np.uint8)
                                    frame[shirt_y:shirt_y+shirt_height, shirt_x:shirt_x+shirt_width] = blended
                            else:  # No alpha channel (fallback)
                                frame[shirt_y:shirt_y+shirt_height, shirt_x:shirt_x+shirt_width] = resized_shirt[:, :, :3]
                        except Exception as e:
                            pass

                # Draw Pose Landmarks (optional - comment out if you don't want skeleton visible)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame_placeholder.image(frame[:, :, ::-1], use_container_width=True)

        cap.release()