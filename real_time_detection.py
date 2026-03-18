import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO
import warnings
import os

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_PATH = 'best_model_21class.pth'
INPUT_VIDEO_PATH = r"C:\Users\stitli\Desktop\Real-Time-Detection\Test_Video.mp4"
OUTPUT_VIDEO_PATH = "Test_video1.mp4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- SMART TUNING ---
CONFIDENCE_THRESHOLD = 0.55     # Moderate threshold
INTERACTION_DISTANCE = 300      # Pixels: If people are closer than this, assume interaction
VIOLENCE_LOCK_FRAMES = 15       # Lock "Fighting" status for 0.5s to prevent flickering
HISTORY_LENGTH = 30

# --- CLASS MAPPING ---
CLASS_MAP = {
    0: 'Vandalism', 1: 'Stealing', 2: 'Shoplifting', 3: 'Shooting',
    4: 'Robbery', 5: 'Roadaccidents', 6: 'Normal', 7: 'Walking',
    8: 'Walking_Phone', 9: 'Walking_Reading', 10: 'Standing', 
    11: 'Sitting', 12: 'Fighting', 13: 'Explosion', 14: 'Meet_Split', 
    15: 'Burglary', 16: 'Clapping', 17: 'Assault', 18: 'Arson', 
    19: 'Arrest', 20: 'Abuse'
}

# LOGIC: If model guesses these, but people are close, CHANGE TO 'FIGHTING'
CONFUSABLE_VIOLENCE = ['Robbery', 'Explosion', 'Assault', 'Abuse', 'Arrest', 'Shooting']
NORMAL_ACTIONS = ['Normal', 'Walking', 'Standing', 'Sitting', 'Clapping']

SKELETON_EDGES = [
    (0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), 
    (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
]

# --- MODEL ARCHITECTURE ---
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else lambda x: x
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))

class STGCN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.net = nn.Sequential(
            STGCNBlock(3, 64),
            STGCNBlock(64, 128),
            STGCNBlock(128, 256),
            STGCNBlock(256, 256)
        )
        self.fcn = nn.Conv1d(256, num_classes, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = self.net(x)
        x = x.mean(dim=3)
        x = x.mean(dim=2, keepdim=True)
        x = self.fcn(x)
        return x.view(N, -1)

# --- SMART ANALYZER ---
class SmartAnalyzer:
    def __init__(self):
        print(f"🚀 Initializing Smart Logic Monitor on {DEVICE}...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.action_model = STGCN(num_classes=21).to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            self.action_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.action_model.eval()
        else:
            print(f"❌ Error: {MODEL_PATH} not found.")
            exit()

        self.history = {}
        self.global_log = set()
        
        # Smart Logic Buffers
        self.track_positions = {}  # {id: (center_x, center_y)}
        self.forced_status = {}    # {id: ("Label", frames_remaining)}
        self.velocity_buffer = {}  # {id: deque([velocities])}
        
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_pose(self, frame):
        # Use simple tracker, we handle ID persistence manually if needed
        results = self.pose_model.track(frame, persist=True, verbose=False, tracker="botsort.yaml", conf=0.3)
        tracks = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            kpts_norm = results[0].keypoints.xyn.cpu().numpy()
            kpts_pixel = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()
            
            for box, id, kp_n, kp_p, conf in zip(boxes, ids, kpts_norm, kpts_pixel, confs):
                model_input = np.hstack([kp_n, conf.reshape(17, 1)])
                
                # Calculate center for distance logic
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                tracks.append({
                    'id': id, 
                    'box': box, 
                    'center': (center_x, center_y),
                    'model_kpts': model_input, 
                    'pixel_kpts': kp_p
                })
        return tracks

    def predict_action(self, skeleton_sequence):
        # Always pad to 30 frames
        seq = np.array(list(skeleton_sequence))
        if len(seq) < 30:
            padding = np.tile(seq[-1], (30 - len(seq), 1, 1))
            seq = np.vstack((seq, padding))
            
        tensor = torch.FloatTensor(seq).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.action_model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, class_idx = torch.max(probs, 1)
        return class_idx.item(), conf.item()

    def apply_smart_logic(self, raw_label, confidence, pid, all_tracks):
        """
        The Brain: Overrides the model's dumb guesses based on logic.
        """
        # 1. LOCK LOGIC: If we are already locked into a fight, stay there
        if pid in self.forced_status and self.forced_status[pid][1] > 0:
            label, timer = self.forced_status[pid]
            self.forced_status[pid] = (label, timer - 1)
            return label, True

        # 2. PROXIMITY LOGIC: Are there other people close by?
        my_pos = self.track_positions.get(pid)
        is_close_interaction = False
        
        if my_pos:
            for other_pid, other_pos in self.track_positions.items():
                if pid != other_pid:
                    dist = self.calculate_distance(my_pos, other_pos)
                    if dist < INTERACTION_DISTANCE:
                        is_close_interaction = True
                        break

        # 3. CORRECTION LOGIC
        final_label = raw_label
        is_violence = False

        if raw_label in CONFUSABLE_VIOLENCE or raw_label == 'Fighting':
            if is_close_interaction:
                # If model thinks it's Robbery/Explosion BUT people are close -> FIGHTING
                final_label = "Fighting"
                is_violence = True
                
                # Lock this decision for 15 frames (0.5s) to stop flickering
                self.forced_status[pid] = ("Fighting", VIOLENCE_LOCK_FRAMES)
            else:
                # If model thinks violence but NO ONE is near -> Likely False Positive
                # Check confidence. If low, suppress it.
                if confidence < 0.65:
                    final_label = "Normal"
                else:
                    # Keep raw label (Maybe it IS a robbery with a gun from distance?)
                    is_violence = True 

        elif raw_label in NORMAL_ACTIONS:
            final_label = raw_label
            is_violence = False

        return final_label, is_violence

    def draw_visuals(self, frame, pid, box, label, is_violence, kpts):
        x1, y1, x2, y2 = map(int, box)
        
        color = (0, 0, 255) if is_violence else (0, 255, 0)
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if not is_violence else 4)
        
        # Skeleton Lines
        for i, j in SKELETON_EDGES:
            if kpts[i][0] > 0 and kpts[j][0] > 0:
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Skeleton Dots
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        # Label
        text = f"ID {pid}" if not is_violence else f"ID {pid}: {label.upper()}!"
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-30), (x1+tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Error: {input_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print(f"🎬 Processing with Logic Layer...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            tracks = self.get_pose(frame)
            
            # Update positions for distance logic
            self.track_positions = {t['id']: t['center'] for t in tracks}
            
            frame_violence = []

            for person in tracks:
                pid = person['id']
                
                # Init history
                if pid not in self.history:
                    self.history[pid] = deque(maxlen=HISTORY_LENGTH)
                
                self.history[pid].append(person['model_kpts'])

                # Predict
                raw_label = "Analyzing..."
                conf = 0.0
                
                # Start predicting EARLY (at 10 frames)
                if len(self.history[pid]) >= 10:
                    idx, conf = self.predict_action(self.history[pid])
                    raw_label = CLASS_MAP.get(idx, "Unknown")

                # --- APPLY SMART LOGIC ---
                final_label, is_violence = self.apply_smart_logic(raw_label, conf, pid, tracks)

                if is_violence:
                    frame_violence.append(final_label)
                    self.global_log.add(final_label)

                self.draw_visuals(frame, pid, person['box'], final_label, is_violence, person['pixel_kpts'])

            # BANNER
            if frame_violence:
                # Prioritize "FIGHTING" in text if multiple labels exist
                display_alert = "FIGHTING" if "Fighting" in frame_violence else frame_violence[0].upper()
                
                cv2.rectangle(frame, (0,0), (w, 60), (0, 0, 255), -1) 
                text = f"!!! SECURITY ALERT: {display_alert} DETECTED !!!"
                text_color = (255, 255, 255)
            else:
                cv2.rectangle(frame, (0,0), (w, 60), (0, 255, 0), -1)
                text = "STATUS: NORMAL CONDITION"
                text_color = (0, 0, 0)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            text_x = (w - tw) // 2
            cv2.putText(frame, text, (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)

            out.write(frame)
            cv2.imshow("Smart Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # REPORT
        print("📊 Generating Report...")
        summary = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(summary, "FINAL ANALYSIS", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        y = 200
        if self.global_log:
            cv2.putText(summary, "INCIDENTS DETECTED:", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for event in self.global_log:
                y += 60
                # Filter out normal stuff from report
                if event not in NORMAL_ACTIONS:
                    cv2.putText(summary, f"- {event}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
            cv2.putText(summary, "NO THREATS DETECTED.", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for _ in range(fps * 5):
            out.write(summary)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Saved: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    analyzer = SmartAnalyzer()
    analyzer.process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)