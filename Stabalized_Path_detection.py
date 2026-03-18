import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- 1. CAMERA CALIBRATION DATA ---
K = np.array([
    [1715.2968, 0, 950.2938],
    [0, 1739.4707, 467.0118],
    [0, 0, 1]
])
dist_coeffs = np.array([-0.975967, 18.773497, -0.071067, 0.003957, -66.4997805])
R = np.array([
    [0.98601396, -0.16237842, 0.03754613],
    [0.03550561, -0.0154512, -0.99925002],
    [0.16283677, 0.98660758, -0.00946975]
])
t = np.array([[-1.75651647], [7.38679332], [24.11008326]])
n_floor = np.array([-9.72394784e-04, 1.03522532e-02, 9.99945941e-01])
d_floor = -0.8481220727825309

# Pre-compute inverses for speed
inv_K = np.linalg.inv(K)
inv_R = R.T
camera_world_pos = (-inv_R @ t).flatten()

# --- 2. HELPER FUNCTIONS ---
def pixel_to_3d_floor(u, v):
    pt_2d = np.array([[[u, v]]], dtype=np.float32)
    # Undistort the point based on lens distortion
    undistorted_pt = cv2.undistortPoints(pt_2d, K, dist_coeffs, P=K)
    u_u, v_u = undistorted_pt[0][0]

    pixel_homo = np.array([u_u, v_u, 1.0])
    ray_cam = inv_K @ pixel_homo
    ray_world = inv_R @ ray_cam
    
    numerator = -(np.dot(n_floor, camera_world_pos) + d_floor)
    denominator = np.dot(n_floor, ray_world)
    
    scale = numerator / denominator
    point_3d = camera_world_pos + scale * ray_world
    return point_3d

# --- 3. INITIALIZE TRACKING ---
model = YOLO('yolov8n.pt') 
video_path = r'C:/Users/stitli/Desktop/Test Video/Stablilize1.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video.")
    exit()

ret, first_frame = cap.read()
reference_frame = first_frame.copy()
path_points_2d = []
path_points_3d = []

# --- 4. MAIN PROCESSING LOOP ---
print("Tracking and Projecting to 3D... Press 'q' to stop.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            # 2D logic: Bottom-center of bounding box (the feet)
            feet_x = int((box[0] + box[2]) / 2)
            feet_y = int(box[3]) 
            
            path_points_2d.append((feet_x, feet_y))
            
            # 3D logic: Project directly to world coordinates
            p3d = pixel_to_3d_floor(feet_x, feet_y)
            path_points_3d.append(p3d)

            # Visual feedback on screen
            cv2.circle(frame, (feet_x, feet_y), 5, (0, 255, 0), -1)

    cv2.imshow("Real-time Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- 5. EXPORT RESULTS ---
output_dir = r'C:/Users/stitli/Desktop/Test Video/'
output_img = os.path.join(output_dir, 'reference_path_result.jpg')
output_ply = os.path.join(output_dir, 'human_path_3d.ply')

# Draw the 2D path on the first frame
for i in range(1, len(path_points_2d)):
    cv2.line(reference_frame, path_points_2d[i-1], path_points_2d[i], (255, 0, 0), 2)
cv2.imwrite(output_img, reference_frame)

# Export the 3D PLY
with open(output_ply, 'w') as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(path_points_3d)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write(f"element edge {len(path_points_3d) - 1}\nproperty int vertex1\nproperty int vertex2\nend_header\n")
    
    for i, p in enumerate(path_points_3d):
        ratio = i / len(path_points_3d)
        r, g = int(255 * ratio), int(255 * (1 - ratio))
        f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} 0\n")
    
    for i in range(len(path_points_3d) - 1):
        f.write(f"{i} {i+1}\n")

print(f"Finished! Saved 2D visual and 3D PLY to {output_dir}")