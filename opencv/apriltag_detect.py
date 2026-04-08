import cv2
import numpy as np
from pupil_apriltags import Detector

detector = Detector(
    families="tag36h11", # common tag family
    nthreads=4, # number of threads to use for detection (optional, default: 1)
    quad_decimate=1.0, # decimate input image by this factor (optional, default: 1.0)
    quad_sigma=0.0, # apply low-pass blur to input (optional, default: 0.0)
    refine_edges=1, # spend more time to align edges of tags (optional, default: 0)
    decode_sharpening=0.25, # apply sharpening to decoded bits (optional, default: 0.25)
)

cap = cv2.VideoCapture(0) # open the default camera

while True:
    ret, frame = cap.read() # read a frame from the camera
    if not ret: 
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale

    # Approximate camera intrinsics for a 640x480 webcam
    # For real use you'd calibrate properly, but this works for testing
    fx = fy = 600.0       # focal length in pixels (rough estimate)
    cx, cy = 320.0, 240.0  # principal point = image center

    camera_params = (fx, fy, cx, cy) # camera parameters for pose estimation
    tag_size = 0.5 # size of the tag in meters (adjust as needed)
    results = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size
        ) # detect AprilTags in the image

    for r in results:
        if r.pose_t is not None:
            tx, ty, tz = r.pose_t.flatten()
            print(f"Tag {r.tag_id} | X: {tx:.3f}m  Y: {ty:.3f}m  Z: {tz:.3f}m")
            # tz = distance from camera to tag, tx and ty are the horizontal and vertical offsets from the camera's optical axis

        # Draw the bounding box around the tag
        corners = r.corners.astype(int) # get the corners of the detected tag and convert to integer
        for i in range(4):
            pt1 = tuple(corners[i]) # start point of the line
            pt2 = tuple(corners[(i + 1) % 4]) # end point of the line (wrap around to the first corner)
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # draw a green line with thickness 2

        # Draw the center point
        cx, cy = int(r.center[0]), int(r.center[1])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) # draw a red circle at the center of the tag

        # Print the tag ID on screen
        cv2.putText(frame, f"ID: {r.tag_id}", (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Tag ID: {r.tag_id} | Center: ({cx}, {cy})")

    cv2.imshow("AprilTag Detection", frame) # display the frame with detected tags

    if cv2.waitKey(1) & 0xFF == ord('q'): # exit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()


