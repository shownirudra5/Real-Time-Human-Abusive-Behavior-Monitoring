import cv2
import numpy as np
import csv
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH       = r'C:/Users/stitli/Desktop/Test Video/Unstable.mp4'
OUTPUT_VIDEO     = r'C:/Users/stitli/Desktop/Test Video/Processed_Output.mp4'
OUTPUT_PLY       = r'C:/Users/stitli/Desktop/Test Video/path_trajectory.ply'
CSV_PATH         = r'C:/Users/stitli/Desktop/Test Video/walking_meta.csv'

FACTOR_X = 8.353 / 1.86  
FACTOR_Y = 27.0 / 6.136  

# Path Aesthetics - BOLDER & BIGGER
TRAJECTORY_COLOR = (255, 100, 0) 
DISC_RADIUS      = 0.15          # Increased for Bolder look
DISC_SAMPLES     = 32
INTERP_STEPS     = 15
Z_LIFT           = 0.05
IMG_W, IMG_H     = 1920, 1080

# ============================================================
# 1. CALIBRATION & GROUND PLANE
# ============================================================
pts_3d_all = np.array([
    [-2.766373, 17.398935, 0.614663], [3.274278, 17.255421, 0.677895],
    [-3.928846, 17.434258, 0.575748], [-4.258451, 3.623070, 0.799107],
    [-4.207079, -0.442319, 0.860467], [-3.077822, -0.716637, 0.618712],
    [-2.960853, 3.743349, 0.762832], [-2.757913, 3.719249, 0.705359],
    [1.435132, 0.270753, 0.814415], [1.633457, -0.856420, 0.890053],
    [1.602883, 11.886833, 0.708620], [4.108355, 12.722070, 0.763674],
    [3.277628, 17.266529, 0.696633], [1.885053, 11.999177, 0.660856],
    [4.160080, 16.811943, 0.681077], [3.653990, -9.094149, 4.792839],
    [3.992291, 10.453816, 4.431476]
], dtype=np.float64)

pts_2d_all = np.array([
    [800, 678], [1042, 680], [754, 678], [676, 815], [640, 889],
    [729, 903], [758, 816], [770, 818], [1033, 879], [1057, 899],
    [993, 723], [1100, 715], [1042, 680], [1001, 723], [1078, 684],
    [1410, 770], [1115, 563]
], dtype=np.float64)

K_opt = np.array([[1700, 0, 960], [0, 1700, 540], [0, 0, 1]], dtype=np.float64)
K_inv = np.linalg.inv(K_opt)
dist  = np.zeros((5,1))

# Initialize Pose
_, rvec_ref, tvec_ref = cv2.solvePnP(pts_3d_all, pts_2d_all, K_opt, dist)

# Ground Plane
floor_pts = pts_3d_all[pts_3d_all[:,2] < 1.2]
centroid = floor_pts.mean(axis=0)
_, _, Vt = np.linalg.svd(floor_pts - centroid)
n_floor = Vt[-1]
if n_floor[2] < 0: n_floor = -n_floor
d_floor = -np.dot(n_floor, centroid)

# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================
def ray_ground(u, v, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).flatten()
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_wld = R.T @ ray_cam
    denom = np.dot(n_floor, ray_wld)
    if abs(denom) < 1e-6: return None
    t = -(np.dot(n_floor, cam_pos) + d_floor) / denom
    return cam_pos + t*ray_wld + n_floor*Z_LIFT

def make_disc(centre, normal, radius, n_s, col):
    v = np.array([1.,0.,0.]) if abs(normal[0])<0.9 else np.array([0.,1.,0.])
    u1 = np.cross(normal,v); u1/=np.linalg.norm(u1)
    u2 = np.cross(normal,u1)
    pts = [(*centre,*col)]
    for i in range(n_s):
        a = 2*np.pi*i/n_s
        pts.append((*(centre+radius*(np.cos(a)*u1+np.sin(a)*u2)),*col))
    return pts

# ============================================================
# 3. PROCESSING LOOP
# ============================================================
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(VIDEO_PATH)
out_video = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (IMG_W, IMG_H))

csv_file = open(CSV_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Distance_Meters', 'X', 'Y', 'Z'])

path_3d = []
total_distance_m = 0.0
primary_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Fix Coordinate Shift: Dynamic SolvePnP per frame
    # (Uses your calibration points to find the camera's exact position in THIS frame)
    _, rvec_curr, tvec_curr = cv2.solvePnP(pts_3d_all, pts_2d_all, K_opt, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    res = model.track(frame, persist=True, classes=[0], verbose=False)
    feet_detected = False

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy().astype(int)
        
        if primary_id is None:
            primary_id = ids[np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))]
        
        if primary_id in ids:
            idx = np.where(ids == primary_id)[0][0]
            x1, y1, x2, y2 = boxes[idx]
            
            if (y2 - y1) / (x2 - x1) > 1.8:
                feet_detected = True
                u_foot, v_foot = (x1 + x2) / 2.0, y2
                # Calculate 3D point using the CURRENT frame camera pose
                p3d = ray_ground(u_foot, v_foot, rvec_curr, tvec_curr)
                
                if p3d is not None:
                    if len(path_3d) > 0:
                        step = np.linalg.norm([
                            (p3d[0] - path_3d[-1][0]) / FACTOR_X,
                            (p3d[1] - path_3d[-1][1]) / FACTOR_Y
                        ])
                        if 0.01 < step < 0.3: total_distance_m += step
                    
                    path_3d.append(p3d)
                    csv_writer.writerow([int(cap.get(cv2.CAP_PROP_POS_FRAMES)), round(total_distance_m, 4), *p3d])

            # Visual Feedback (Kept exactly as requested)
            color = (0, 255, 0) if feet_detected else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (int((x1+x2)/2), int(y2)), 10, (0, 165, 255), -1) # Point is bigger
            
            cv2.rectangle(frame, (0, 0), (550, 110), (0,0,0), -1)
            cv2.putText(frame, f"DIST: {total_distance_m:.2f} m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "FEET DETECTED" if feet_detected else "PAUSED", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_video.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# ============================================================
# 4. EXPORT BOLD 3D PATH
# ============================================================
all_verts = []
for i in range(len(path_3d)):
    # Bolder discs for every point
    all_verts.extend(make_disc(path_3d[i], n_floor, DISC_RADIUS, DISC_SAMPLES, TRAJECTORY_COLOR))
    if i < len(path_3d)-1:
        for s in range(1, INTERP_STEPS):
            p = (1 - s/INTERP_STEPS)*path_3d[i] + (s/INTERP_STEPS)*path_3d[i+1]
            # Interpolation is now full width to make a solid bold tube
            all_verts.extend(make_disc(p, n_floor, DISC_RADIUS, 16, TRAJECTORY_COLOR))

with open(OUTPUT_PLY, 'w') as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(all_verts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for v in all_verts: f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(v[3])} {int(v[4])} {int(v[5])}\n")

cap.release()
out_video.release()
csv_file.close()
cv2.destroyAllWindows()