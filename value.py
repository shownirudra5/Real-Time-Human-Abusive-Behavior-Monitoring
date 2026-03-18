import cv2
import numpy as np

# 1. YOUR 12 MATCHING POINTS (2D Pixels and 3D CloudCompare)
# 2D points from your video frame
pts_2d = np.array([
    [657,345], [906, 354], [657,726], [898, 727], [945, 370], 
    [970, 360], [1394, 199], [1654, 92], [510, 246], [545, 302], 
    [539, 863], [514, 935]
], dtype=np.float32)

# 3D points from CloudCompare (RealityScan model)
pts_3d = np.array([
    [-2.845743, 17.492287, 9.812160], [3.110748, 17.288469, 9.905505],
    [-2.763717, 17.454931, 0.664997], [3.269407, 17.2482660, 0.672732],
    [4.079510, 16.818336, 9.568498], [4.039400, 12.892061, 9.548595],
    [3.650068, -9.618960, 9.852505], [4.73447, -13.073077, 10.377557],
    [-4.113413, -4.984396, 9.51431], [-4.387230, 3.769322, 9.616863],
    [-4.238680, 3.652949, 0.805447], [-4.199863, -0.412505, 0.848958]
], dtype=np.float32)

# 2. DEFINE IMAGE DIMENSIONS (1080p)
width, height = 1920, 1080
cx_init, cy_init = width / 2, height / 2

# 3. INITIAL GUESS FOR INTRINSICS (From your COLMAP log)
# We use your 2304 focal length as a starting point for the optimizer
initial_K = np.array([
    [2304.0, 0, cx_init],
    [0, 2304.0, cy_init],
    [0, 0, 1]
], dtype=np.float32)

# 4. PERFORM CALIBRATION
# This calculates K, Distortion, and Extrinsics simultaneously
# 'ret' is the RMS error (smaller is better, aim for < 1.0)
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    [pts_3d], [pts_2d], (width, height), initial_K, None, 
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
)

# 5. CONVERT ROTATION VECTOR TO MATRIX
R_matrix, _ = cv2.Rodrigues(rvecs[0])
T_vector = tvecs[0]

# 6. CALCULATE ACTUAL TRIPOD POSITION (Camera Center in 3D)
# Formula: C = -R^T * t
camera_pos = -R_matrix.T @ T_vector

# --- OUTPUT RESULTS ---
print("="*30)
print("1. CAMERA INTRINSICS (K)")
print(f"   fx: {K[0,0]:.4f}")
print(f"   fy: {K[1,1]:.4f}")
print(f"   cx: {K[0,2]:.4f}")
print(f"   cy: {K[1,2]:.4f}")

print("\n2. DISTORTION PARAMETERS")
print(f"   k1: {dist_coeffs[0][0]:.6f}")
print(f"   k2: {dist_coeffs[0][1]:.6f}")
print(f"   p1: {dist_coeffs[0][2]:.6f}")
print(f"   p2: {dist_coeffs[0][3]:.6f}")

print("\n3. CAMERA EXTRINSICS")
print("   Rotation Matrix (R):")
print(R_matrix)
print(f"\n   Translation Vector (t):")
print(T_vector.flatten())

print("\n4. WORLD POSITION (Tripod Location)")
print(f"   X: {camera_pos[0][0]:.4f}")
print(f"   Y: {camera_pos[1][0]:.4f}")
print(f"   Z: {camera_pos[2][0]:.4f}")
print("="*30)
print(f"RMS Re-projection Error: {ret:.4f} pixels")