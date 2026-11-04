import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam is working properly. Press 'q' to exit.")

while True:
    #Capture frame
    success, frame = cap.read()

    if not success:
        print("Failed to capture frame. Exiting...")
        break

    # Mirror the webcam feed
    frame = cv2.flip(frame, 1)

    # Display video
    cv2.imshow("AI Virtual Whiteboard - Webcam", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 