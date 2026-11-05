import cv2
import mediapipe as mp
import numpy as np

from gesture_recognition import detect_fingers, recognize_gesture
from drawing import draw_line, clear_canvas

# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    exit()

# Get webcam dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create blank drawing canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Drawing variables
previous_point = None
drawing_color = (0, 255, 0)
brush_thickness = 7

previous_point = None

drawing_color = (0, 255, 0)  # Green (BGR)
color_name = "Green"

brush_thickness = 7
is_erasing = False
eraser_thickness = 35

print("Virtual Whiteboard Started!")
print("Index finger: Draw")
print("Index + Middle fingers: Pause")
print("Press Q to exit")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                cv2.getTickCount() /
                cv2.getTickFrequency() * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:

                for hand_landmarks in result.hand_landmarks:

                    points = []

                    for landmark in hand_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        points.append((x, y))

                    # Draw hand skeleton
                    for start, end in HAND_CONNECTIONS:

                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # Detect fingers
                    fingers = detect_fingers(hand_landmarks)

                    gesture = recognize_gesture(fingers)

                    # Get index fingertip
                    current_point = points[8]

                    # Drawing mode
                    # if gesture == "DRAW":

                    #     if previous_point is not None:

                    #         draw_line(
                    #             canvas,
                    #             previous_point,
                    #             current_point,
                    #             drawing_color,
                    #             brush_thickness
                    #         )

                    #     previous_point = current_point
                    

                    # else:
                    #     # Stop drawing
                    #     previous_point = None

                    if gesture == "DRAW":

                        active_color = (0, 0, 0) if is_erasing else drawing_color
                        active_thickness = eraser_thickness if is_erasing else brush_thickness

                        if previous_point is not None:
                            draw_line(
                                canvas,
                                previous_point,
                                current_point,
                                active_color,
                                active_thickness
                            )

                        previous_point = current_point

                    else:
                        previous_point = None

                    # Highlight index fingertip
                    cv2.circle(
                        frame,
                        current_point,
                        10,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

            else:
                previous_point = None

            tool_name = "ERASER" if is_erasing else "BRUSH"

            cv2.putText(
                frame,
                f"Tool: {tool_name} | Color: {color_name} | Size: {brush_thickness}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "1: Green  2: Blue  3: Red  4: Yellow",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "E: Eraser  C: Clear  +/-: Brush Size  Q: Quit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )
            # Combine webcam and drawing canvas
            output = cv2.add(frame, canvas)

            cv2.imshow(
                "AI Virtual Whiteboard",
                output
            )

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord('q'):
                break

            # Select colors
            elif key == ord('1'):
                drawing_color = (0, 255, 0)
                color_name = "Green"
                is_erasing = False

            elif key == ord('2'):
                drawing_color = (255, 0, 0)
                color_name = "Blue"
                is_erasing = False

            elif key == ord('3'):
                drawing_color = (0, 0, 255)
                color_name = "Red"
                is_erasing = False

            elif key == ord('4'):
                drawing_color = (0, 255, 255)
                color_name = "Yellow"
                is_erasing = False

            # Toggle eraser
            elif key == ord('e'):
                is_erasing = not is_erasing

            # Clear canvas
            elif key == ord('c'):
                clear_canvas(canvas)
                previous_point = None

            # Increase brush size
            elif key in (ord('+'), ord('=')):
                brush_thickness = min(50, brush_thickness + 2)

            # Decrease brush size
            elif key in (ord('-'), ord('_')):
                brush_thickness = max(2, brush_thickness - 2)

finally:
    cap.release()
    cv2.destroyAllWindows()
