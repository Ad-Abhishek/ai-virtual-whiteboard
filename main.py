import cv2
import mediapipe as mp

# Load MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/hand_landmarker.task"

# Configure hand detection
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hand skeleton connections
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

print("Hand detection started. Press Q to exit.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # Get frame dimensions
            height, width, _ = frame.shape

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate timestamp in milliseconds
            timestamp_ms = int(
                cv2.getTickCount() /
                cv2.getTickFrequency() * 1000
            )

            # Detect hand landmarks
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Process detected hands
            if result.hand_landmarks:

                for hand_landmarks in result.hand_landmarks:

                    # Convert normalized coordinates to pixels
                    points = []

                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    # Draw connections
                    for start, end in HAND_CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # Draw landmark points
                    for point in points:
                        cv2.circle(
                            frame,
                            point,
                            5,
                            (0, 0, 255),
                            -1
                        )

                    # Highlight index fingertip
                    index_tip = points[8]

                    cv2.circle(
                        frame,
                        index_tip,
                        10,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        "Hand Detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            else:
                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Display output
            cv2.imshow("AI Virtual Whiteboard - Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
