import streamlit as st
import cv2
import tempfile
import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path

# ------------------ Configuration ------------------
MODEL_PATH = r"C:\Users\rashi\Desktop\LPU\Capstone\best2_retrained_v2.pt"
LSTM_MODEL_PATH = "C:/Users/rashi/Desktop/LPU/Capstone/animal_behaviour_dataset/lstm_model.pth"
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Define the LSTM Model ------------------
class BehaviorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(BehaviorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# ------------------ Streamlit UI Setup ------------------
def setup_ui():
    # Set page configuration
    st.set_page_config(
        page_title="Wildlife Protection System",
        page_icon="🪲",
        layout="wide",
    )
    
    # Apply custom CSS with simpler, more compatible styling
    st.markdown("""
    <style>
        h1 {
            color: #2E7D32; 
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 2px solid #4CAF50;
            margin-bottom: 20px;
        }
        h2, h3 {
            color: #2E7D32;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        .info-box {
            background-color: #E8F5E9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #4CAF50;
        }
        .warning-box {
            background-color: #FFF8E1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #FFC107;
        }
        .danger-box {
            background-color: #FFEBEE;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 5px solid #F44336;
        }
        .upload-container {
            background-color: #F9FFF9;
            padding: 20px;
            border-radius: 10px;
            border: 1px dashed #4CAF50;
            margin-bottom: 25px;
        }
        .section-container {
            background-color: #F5F5F5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 14px;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------ Main Application ------------------
def main():
    # Setup UI styling
    setup_ui()
    
    # Header section
    st.markdown("<h1>Wildlife Protection & Poaching Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Advanced computer vision system to detect poaching activities and analyze animal behavior patterns</div>", unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Main upload section
        st.markdown("<h3>Upload Wildlife Footage</h3>", unsafe_allow_html=True)
        st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Upload a video to detect poaching-related objects and analyze animal behavior",
                                         type=["mp4", "avi", "mov", "mkv", "mpeg4"])
        st.caption("Supports MP4, AVI, MOV, MKV, MPEG4 formats")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # System information
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("<h3>System Capabilities</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Object Detection**: Identifies animals, poachers
        - **Behavior Analysis**: Detects unusual animal movements
        - **Threat Assessment**: Evaluates poaching likelihood
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Technology stack info
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("<h3>Technology</h3>", unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].metric("Model", "YOLOv8")
        cols[1].metric("Analysis", "LSTM")
        cols[2].metric("Mode", "Video")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Load models and process video if uploaded
    if uploaded_video:
        try:
            # Load the YOLO model
            model = YOLO(MODEL_PATH)
            class_names = model.names
            
            # Load the LSTM model
            lstm_model = BehaviorLSTM()
            lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=torch.device('cpu')))
            lstm_model.eval()
            
            # Process video
            process_video(uploaded_video, model, lstm_model, class_names)
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
    else:
        # Show information when no video is uploaded
        with st.expander("ℹ️ How This System Works", expanded=True):
            st.markdown("""
            ### Detection Process
            1. Upload wildlife footage from surveillance cameras or drones
            2. The system analyzes each frame using advanced AI models
            3. Objects like animals, humans, and equipment are detected and tracked
            4. Animal behavior patterns are analyzed for signs of distress
            5. Potential poaching threats are identified and highlighted
            
            ### Interpreting Results
            - **Normal Behavior**: Regular movement patterns with no detected threats
            - **Abnormal Behavior**: Unusual animal reactions that may indicate threats
            - **Poaching Detection**: Direct identification of poachers or equipment
            """)
    
    # Footer
    st.markdown("<div class='footer'>Wildlife Protection & Poaching Detection System v2.0<br>© 2025 Wildlife Conservation Initiative</div>", unsafe_allow_html=True)

# Define poaching-related classes
POACHING_CLASSES = [26]  # "Poacher" class

# ------------------ Video Processing Function ------------------
def process_video(uploaded_video, model, lstm_model, class_names):
    # Create temporary file
    temp_input_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input_video.write(uploaded_video.read())
    temp_input_video_path = temp_input_video.name
    
    # Setup progress tracking
    st.markdown("<h3>Processing Video</h3>", unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("Initializing video processing...")
    
    # Open video file
    cap = cv2.VideoCapture(temp_input_video_path)
    if not cap.isOpened():
        st.error("Failed to open the video file. Please check if the file is valid.")
        os.remove(temp_input_video_path)
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_output_path = os.path.join(OUTPUT_DIR, "processed_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error("Failed to initialize video writer. Please check system permissions.")
        cap.release()
        os.remove(temp_input_video_path)
        return
    
    # Initialize tracking variables
    prev_boxes, prev_ids, prev_classes = None, None, None
    time_delta = 1 / fps
    sequences_by_id = {}  # Track sequences per ID
    seq_length = 100
    downsample_factor = 5
    frame_count = 0
    
    detected_objects = set()
    behavior_states = []
    poaching_detected = False
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Update progress
        progress = min(int((frame_count / total_frames) * 100), 100)
        progress_bar.progress(progress)
        if frame_count % 30 == 0:
            status_text.info(f"Processing frame {frame_count}/{total_frames} ({progress}%)")
        
        # Dynamic confidence threshold
        conf_threshold = 0.2 if frame_count < 50 else 0.4
        results = model.track(frame, persist=True, verbose=False, conf=conf_threshold, iou=0.6)
        
        if not results or not results[0].boxes or results[0].boxes.id is None:
            out.write(frame)
            continue
        
        curr_boxes = results[0].boxes.xyxy.cpu().numpy()
        curr_ids = results[0].boxes.id.cpu().numpy()
        curr_classes = results[0].boxes.cls.cpu().numpy()
        curr_confs = results[0].boxes.conf.cpu().numpy()
        
        # Draw bounding boxes and labels
        for box, cls, conf in zip(curr_boxes, curr_classes, curr_confs):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            label = f"{class_names[cls]} {conf:.2f}"
            detected_objects.add(class_names[cls])
            
            # Different colors for different object types
            if cls in POACHING_CLASSES:
                color = (0, 0, 255)  # Red for poachers
                poaching_detected = True
            elif "animal" in class_names[cls].lower():
                color = (0, 255, 0)  # Green for animals
            else:
                color = (255, 165, 0)  # Orange for other objects
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add a background behind the text for better visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate features for LSTM
        if prev_boxes is not None and frame_count % downsample_factor == 0:
            for curr_id, curr_box, curr_class in zip(curr_ids, curr_boxes, curr_classes):
                curr_id = int(curr_id)
                if curr_id in prev_ids and curr_class != 26:
                    prev_idx = list(prev_ids).index(curr_id)
                    prev_box = prev_boxes[prev_idx]
                    prev_x, prev_y = (prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2
                    curr_x, curr_y = (curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    speed = distance / time_delta if time_delta > 0 else 0
                    direction = np.arctan2(curr_y - prev_y, curr_x - prev_x) * 180 / np.pi
                    
                    if curr_id not in sequences_by_id:
                        sequences_by_id[curr_id] = []
                    sequences_by_id[curr_id].append([speed, direction])
        
        prev_boxes, prev_ids, prev_classes = curr_boxes, curr_ids, curr_classes
        
        # Predict behavior with LSTM
        for curr_id, seq in list(sequences_by_id.items()):
            if len(seq) >= seq_length:
                X = torch.FloatTensor(np.array([seq[-seq_length:]]))
                with torch.no_grad():
                    pred = lstm_model(X).item()
                    behavior_states.append(pred)
                sequences_by_id[curr_id] = []  # Clear sequence after prediction
        
        # Determine behavior state and threat likelihood
        behavior_state = "Normal"
        threat_likelihood = "Low"
        if poaching_detected:
            behavior_state = "Not Normal (Poaching Detected)"
            threat_likelihood = "High"
        elif behavior_states and any(pred > 0.5 for pred in behavior_states):
            behavior_state = "Not Normal (Behavior Deviation)"
            threat_likelihood = "Moderate"
        
        # Create a semi-transparent overlay for status information
        overlay = frame.copy()
        cv2.rectangle(overlay, (30, 30), (500, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add status text with colored indicators
        behavior_color = (0, 255, 0) if behavior_state == "Normal" else (0, 0, 255)
        threat_color = (0, 255, 0) if threat_likelihood == "Low" else (0, 165, 255) if threat_likelihood == "Moderate" else (0, 0, 255)
        
        cv2.putText(frame, f"Behavior: {behavior_state}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, behavior_color, 2)
        cv2.putText(frame, f"Threat Likelihood: {threat_likelihood}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, threat_color, 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    # Clean up video processing
    cap.release()
    out.release()
    status_text.success("Video processing complete!")
    progress_bar.progress(100)
    
    # Convert to playable MP4 using FFmpeg
    final_output_path = os.path.join(OUTPUT_DIR, "processed_final.mp4")
    ffmpeg_command = ["ffmpeg", "-y", "-i", raw_output_path, "-vcodec", "libx264", "-acodec", "aac", final_output_path]
    
    try:
        st.info("Finalizing video output...")
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        
        # Display results
        st.markdown("<h3>Analysis Results</h3>", unsafe_allow_html=True)
        
        # Status display with appropriate styling
        status_class = "info-box"
        if "Not Normal" in behavior_state:
            if "Poaching Detected" in behavior_state:
                status_class = "danger-box"
            else:
                status_class = "warning-box"
                
        st.markdown(f"<div class='{status_class}'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Detected Objects")
            st.write(", ".join(detected_objects) if detected_objects else "No objects detected.")
            
        with col2:
            st.subheader("Behavior Analysis")
            st.write(f"**Status:** {behavior_state}")
            
        with col3:
            st.subheader("Threat Assessment")
            if threat_likelihood == "Low":
                st.write("**Threat Level:** Low ✅")
            elif threat_likelihood == "Moderate":
                st.write("**Threat Level:** Moderate ⚠️")
            else:
                st.write("**Threat Level:** High 🚨")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display processed video
        st.markdown("<h3>Processed Video</h3>", unsafe_allow_html=True)
        st.video(final_output_path)
        
        # Download button
        with open(final_output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="wildlife_protection_analysis.mp4",
                mime="video/mp4"
            )
            
    except subprocess.CalledProcessError as e:
        st.error(f"Video finalization failed: {e.stderr}")
    
    # Clean up temporary files
    for file_path in [temp_input_video_path, raw_output_path, final_output_path]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file: {e}")

# Run the application
if __name__ == "__main__":
    main()