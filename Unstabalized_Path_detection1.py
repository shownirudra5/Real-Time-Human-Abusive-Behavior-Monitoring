import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH   = r'C:/Users/stitli/Desktop/Test Video/Unstable.mp4'
OUTPUT_PLY   = r'C:/Users/stitli/Desktop/Test Video/path_trajectory.ply'

TRAJECTORY_COLOR = (255, 100, 0)   # R,G,B  — single orange colour
DISC_RADIUS      = 0.05            # metres — size of each dot in 3D viewer
DISC_SAMPLES     = 20              # points per disc
INTERP_STEPS     = 10             # sub-points between frames for solid line
Z_LIFT           = 0.05           # metres above ground so path sits ON mesh

# ============================================================
# 1. KNOWN 2D ↔ 3D CORRESPONDENCES FROM FRAME 1
# ============================================================
pts_3d_all = np.array([
    [2.837687,  17.3496449, 9.822531],   # 0
    [3.128246,  17.301996,  9.878275],   # 1
    [-2.766373, 17.398935,  0.614663],   # 2
    [3.274278,  17.255421,  0.677895],   # 3
    [4.095757,  12.831632,  0.68746 ],   # 4
    [-3.928846, 17.434258,  0.575748],   # 5
    [-4.258451,  3.623070,  0.799107],   # 6
    [-4.207079, -0.442319,  0.860467],   # 7
    [3.653990,  -9.094149,  4.792839],   # 8
    [3.649345,  -9.58773,   9.850519],   # 9
    [-3.077822, -0.716637,  0.618712],   # 10
    [-2.960853,  3.743349,  0.762832],   # 11
    [-2.757913,  3.719249,  0.705359],   # 12
    [1.435132,   0.270753,  0.814415],   # 13
    [1.633457,  -0.856420,  0.890053],   # 14
    [1.602883,  11.886833,  0.708620],   # 15
    [3.992291,  10.453816,  4.431476]    # 16
], dtype=np.float64)

pts_2d_all = np.array([
    [803, 305],   # 0
    [1048, 310],  # 1
    [800, 678],   # 2
    [1042, 680],  # 3
    [1100, 715],  # 4
    [754, 678],   # 5
    [676, 815],   # 6
    [640, 889],   # 7
    [1410, 770],  # 8
    [1476, 178],  # 9
    [729, 903],   # 10
    [758, 816],   # 11
    [770, 818],   # 12
    [1033, 879],  # 13
    [1057, 899],  # 14
    [993, 723],   # 15
    [1115, 563]   # 16
], dtype=np.float64)

# ============================================================
# 2. CALIBRATE CAMERA FROM FRAME 1
#    Strategy: optimise focal length + cx, cy together with
#    pose via bundle-adjustment style minimisation so that
#    reprojection error is truly minimised.
# ============================================================
IMG_W, IMG_H = 1920, 1080

def build_K(fx, fy, cx, cy):
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]], dtype=np.float64)

def reprojection_residuals(params, pts3d, pts2d):
    """params = [fx, fy, cx, cy, rx, ry, rz, tx, ty, tz]"""
    fx, fy, cx_, cy_ = params[0], params[1], params[2], params[3]
    rvec = params[4:7].reshape(3,1)
    tvec = params[7:10].reshape(3,1)
    K_   = build_K(fx, fy, cx_, cy_)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K_, np.zeros(5))
    diff = pts2d.reshape(-1,2) - proj.reshape(-1,2)
    return diff.flatten()

def reprojection_cost(params, pts3d, pts2d):
    r = reprojection_residuals(params, pts3d, pts2d)
    return np.dot(r, r)

# Initial guess with solvePnP
K0   = build_K(1700., 1700., IMG_W/2., IMG_H/2.)
dist = np.zeros((5,1), dtype=np.float64)
ok, rvec0, tvec0 = cv2.solvePnP(pts_3d_all, pts_2d_all, K0, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
rvec0, tvec0 = cv2.solvePnPRefineLM(pts_3d_all, pts_2d_all, K0, dist, rvec0, tvec0)

print("=== INITIAL REPROJECTION ERRORS ===")
proj0, _ = cv2.projectPoints(pts_3d_all, rvec0, tvec0, K0, dist)
errs0 = np.linalg.norm(pts_2d_all - proj0.reshape(-1,2), axis=1)
for i,(e,p2,p3) in enumerate(zip(errs0, pts_2d_all, pts_3d_all)):
    print(f"  pt{i:2d}  err={e:6.2f}px  2d=({p2[0]:.0f},{p2[1]:.0f})  "
          f"3d=({p3[0]:.2f},{p3[1]:.2f},{p3[2]:.2f})")
print(f"  MEAN={errs0.mean():.2f}  MAX={errs0.max():.2f}\n")

# Bundle-adjust: optimise fx,fy,cx,cy + pose together
x0 = np.array([1700., 1700., IMG_W/2., IMG_H/2.,
                rvec0[0,0], rvec0[1,0], rvec0[2,0],
                tvec0[0,0], tvec0[1,0], tvec0[2,0]])

bounds = [(800, 3000), (800, 3000),           # fx, fy
          (IMG_W*0.3, IMG_W*0.7),             # cx
          (IMG_H*0.3, IMG_H*0.7),             # cy
          (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),  # rvec
          (-50, 50), (-50, 50), (-50, 50)]    # tvec

res = minimize(reprojection_cost, x0,
               args=(pts_3d_all, pts_2d_all),
               method='L-BFGS-B', bounds=bounds,
               options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10})

fx_opt, fy_opt, cx_opt, cy_opt = res.x[0], res.x[1], res.x[2], res.x[3]
rvec_opt = res.x[4:7].reshape(3,1)
tvec_opt = res.x[7:10].reshape(3,1)
K_opt    = build_K(fx_opt, fy_opt, cx_opt, cy_opt)
R_opt, _ = cv2.Rodrigues(rvec_opt)
cam_world = (-R_opt.T @ tvec_opt).flatten()

print("=== OPTIMISED REPROJECTION ERRORS ===")
proj_opt, _ = cv2.projectPoints(pts_3d_all, rvec_opt, tvec_opt, K_opt, dist)
errs_opt = np.linalg.norm(pts_2d_all - proj_opt.reshape(-1,2), axis=1)
for i, e in enumerate(errs_opt):
    flag = "  ← HIGH (check 3D/2D match!)" if e > 8 else ""
    print(f"  pt{i:2d}  err={e:6.2f}px{flag}")
print(f"  MEAN={errs_opt.mean():.2f}  MAX={errs_opt.max():.2f}")
print(f"  fx={fx_opt:.1f}  fy={fy_opt:.1f}  cx={cx_opt:.1f}  cy={cy_opt:.1f}\n")

# ============================================================
# 3. FIT GROUND PLANE FROM FLOOR-LEVEL 3D POINTS
#    Only use points that are actually on the floor (Z ~ 0.6-0.9 m)
# ============================================================
# Select points whose Z coord is in the floor range
floor_mask = (pts_3d_all[:, 2] < 1.5)   # exclude pts 0,1,8,9,16 (Z>4)
floor_pts  = pts_3d_all[floor_mask]

centroid  = floor_pts.mean(axis=0)
_, _, Vt  = np.linalg.svd(floor_pts - centroid)
n_floor   = Vt[-1]
if n_floor[2] < 0:
    n_floor = -n_floor
d_floor   = -np.dot(n_floor, centroid)

print(f"Ground plane normal: {n_floor.round(5)}")
print(f"Ground plane d:      {d_floor:.5f}")
print(f"Using {floor_mask.sum()} floor points for plane fit\n")

# ============================================================
# 4. RAY → GROUND INTERSECTION
# ============================================================
K_inv = np.linalg.inv(K_opt)

def ray_ground_intersection(u, v):
    """Back-project pixel (u,v) in Frame-1 space → 3D ground point."""
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_wld = R_opt.T @ ray_cam
    denom   = np.dot(n_floor, ray_wld)
    if abs(denom) < 1e-6:
        return None
    t = -(np.dot(n_floor, cam_world) + d_floor) / denom
    if t < 0:
        return None
    p = cam_world + t * ray_wld
    p = p + n_floor * Z_LIFT          # lift above mesh surface
    return p

# ============================================================
# 5. VIDEO PROCESSING  (frame-by-frame stabilisation → 3D path)
# ============================================================
model = YOLO('yolov8n.pt')
cap   = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open: {VIDEO_PATH}"

ret, first_frame = cap.read()
assert ret, "Cannot read first frame"

# ORB for homography stabilisation back to Frame 1
orb      = cv2.ORB_create(4000)
gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

raw_path  = []   # list of np.array(3,)
frame_idx = 0
print("Processing video… press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # ── Homography: current frame → Frame 1 pixel space ──────────────────
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)
    H = None
    if des_curr is not None and len(des_curr) > 10:
        matches = sorted(bf.match(des_ref, des_curr),
                         key=lambda x: x.distance)[:300]
        if len(matches) >= 8:
            src = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 3.0,
                                          maxIters=3000, confidence=0.999)

    # ── YOLO: detect human foot ───────────────────────────────────────────
    results = model.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes.id is not None and H is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            u_curr = (box[0] + box[2]) / 2.0
            v_curr = float(box[3])

            # Warp foot pixel into Frame-1 reference space
            pt_warped = cv2.perspectiveTransform(
                np.array([[[u_curr, v_curr]]], dtype=np.float32), H)[0][0]
            u_ref, v_ref = float(pt_warped[0]), float(pt_warped[1])

            # Only accept points inside image bounds
            if not (0 <= u_ref < IMG_W and 0 <= v_ref < IMG_H):
                continue

            p3d = ray_ground_intersection(u_ref, v_ref)
            if p3d is not None:
                raw_path.append(p3d)

    if frame_idx % 100 == 0:
        print(f"  frame {frame_idx}  →  {len(raw_path)} path pts")

    cv2.imshow("Processing (q=quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal raw path points: {len(raw_path)}")

# ============================================================
# 6. BUILD TRAJECTORY PLY
#    Each raw point → large disc (visible in CloudCompare)
#    Between consecutive points → interpolated smaller discs (solid line)
#    All points = single colour defined by TRAJECTORY_COLOR
# ============================================================
def make_disc(centre, normal, radius, n_samples, color_rgb):
    v  = np.array([1.,0.,0.]) if abs(normal[0]) < 0.9 else np.array([0.,1.,0.])
    u1 = np.cross(normal, v);  u1 /= np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    pts = [(*centre, *color_rgb)]
    for i in range(n_samples):
        angle = 2 * np.pi * i / n_samples
        p = centre + radius * (np.cos(angle)*u1 + np.sin(angle)*u2)
        pts.append((*p, *color_rgb))
    return pts

n_raw       = len(raw_path)
all_verts   = []
col         = TRAJECTORY_COLOR   # single colour for whole trajectory

print(f"Building PLY discs…")
for i in range(n_raw):
    # Main disc at detection point
    all_verts.extend(make_disc(raw_path[i], n_floor, DISC_RADIUS, DISC_SAMPLES, col))

    # Interpolated tube between this and next point
    if i < n_raw - 1:
        for s in range(1, INTERP_STEPS):
            alpha   = s / INTERP_STEPS
            p_interp = (1 - alpha) * raw_path[i] + alpha * raw_path[i+1]
            # Smaller disc for the connecting tube
            all_verts.extend(make_disc(p_interp, n_floor,
                                       DISC_RADIUS * 0.5,
                                       DISC_SAMPLES // 2, col))

print(f"Total PLY vertices: {len(all_verts)}")

with open(OUTPUT_PLY, 'w') as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(all_verts)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for v in all_verts:
        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                f"{int(v[3])} {int(v[4])} {int(v[5])}\n")

print(f"\n✓ PLY saved → {OUTPUT_PLY}")
print(f"\nIn CloudCompare:")
print(f"  • File → Open → load PLY  (it will appear as coloured points)")
print(f"  • Edit → Point size → 6-8 px for best visibility")
print(f"  • The path is {Z_LIFT*100:.0f} cm above ground — increase Z_LIFT")
print(f"    at top of script if it still clips into the mesh")
print(f"\nIf reprojection errors for specific points are HIGH (>8px),")
print(f"those 2D↔3D matches need re-checking in Frame 1 of the video.")q