# Models

`face_detection_yunet_2023mar.onnx` is the YuNet face detector from OpenCV Zoo
(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), MIT License; see that page
for the full licence text. `framing.py` uses it through OpenCV's `cv2.FaceDetectorYN` to follow speakers in
the crop/auto layouts and to check or find a streamer's facecam. If the file is missing, OpenCV's built-in
Haar cascade is used instead (less accurate).
