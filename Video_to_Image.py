import cv2
import os

# 1. Setup paths
video_path = r'C:/Users/stitli/Desktop/Test Video/Unstable.mp4'
output_dir = r'C:/Users/stitli/Desktop/Test Video/'
output_name = 'reference_frame_0.jpg'

# 2. Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
else:
    # 3. Read the first frame
    ret, frame = cap.read()
    
    if ret:
        # Save the frame
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, frame)
        
        print("-" * 30)
        print(f"SUCCESS!")
        print(f"Frame saved to: {output_path}")
        print("-" * 30)
        print("INSTRUCTIONS:")
        print("1. Open this image in an editor (like Paint or Photoshop).")
        print("2. Find at least 4 points on the floor (corners of tiles, rug edges, etc.).")
        print("3. Note the (x, y) pixel coordinates for each.")
        print("4. Measure the real-world (X, Y, Z) distance between those points.")
    else:
        print("Error: Could not read the first frame. The file might be corrupted.")

cap.release()