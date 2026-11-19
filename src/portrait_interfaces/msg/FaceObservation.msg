# One face detection, produced by portrait_perception for every processed camera frame.
std_msgs/Header header          # stamp of the image, frame_id = camera optical frame
bool detected
float32 score
# Pixel geometry in the full-resolution image
float32[4] bbox                 # x, y, w, h
float32[10] landmarks           # right eye, left eye, nose tip, right mouth, left mouth (x, y)
uint32 image_width
uint32 image_height
# Head pose from PnP with a generic face model (camera optical frame)
geometry_msgs/Point position_camera   # face centre (between the eyes) [m]
float32 yaw_deg                 # + = subject turned to their left (image right)
float32 pitch_deg               # + = chin up
float32 roll_deg
# Face centre in the robot base frame, using the camera pose at capture time
bool has_base_position
geometry_msgs/Point position_base
float32 sharpness               # variance of Laplacian inside the face box
geometry_msgs/Pose camera_pose  # camera optical frame in the base frame at the image stamp (if has_base_position)
bool complete                   # whole face box inside the image (reliable for measurement)
