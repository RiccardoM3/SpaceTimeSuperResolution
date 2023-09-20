import cv2
import time

def crop(frame):
    # Get the height and width of the frame
    H, W, _ = frame.shape

    desired_aspect_ratio = 384 / 256
    current_aspect_ratio = W / H

    # Decide whether to crop from the top/bottom or left/right
    if current_aspect_ratio > desired_aspect_ratio:
        new_width = int(H * desired_aspect_ratio)
        start_x = (W - new_width) // 2
        cropped_frame = frame[:, start_x:start_x + new_width]
    else:
        new_height = int(W / desired_aspect_ratio)
        start_y = (H - new_height) // 2
        cropped_frame = frame[start_y:start_y + new_height, :]

    return cropped_frame

def resize(frame, size):
    return cv2.resize(frame, size, interpolation="INTER_AREA")

# Open the video stream (you can replace 0 with your desired video file name)
video_stream = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_time = 0
frame_count = 0

context = []

while True:
    ret, frame = video_stream.read()
    if not ret:
        break

    cropped_frame = crop(frame)
    ground_truth = resize(cropped_frame, (384, 256))
    model_input = resize(ground_truth, (384/4, 256/4))
    
    context.append(model_input)
    if len(context) > 4:
        context.pop(0)
    
    cv2.imshow('Video', cropped_frame)

    # Calculate FPS
    current_time = time.time()
    frame_count += 1
    if (current_time - prev_time) > 1:
        fps = frame_count / (current_time - prev_time)
        print(f"FPS: {fps:.2f}")
        prev_time = current_time
        frame_count = 0

    # Check for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_stream.release()
cv2.destroyAllWindows()