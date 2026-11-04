def detect_fingers(hand_landmarks):
    """
    Detect which of the four fingers are raised.

    Returns:
        A list containing 1 (raised) or 0 (folded).
        Order: Index, Middle, Ring, Pinky
    """

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers = []

    for tip, pip in zip(finger_tips, finger_pips):

        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def recognize_gesture(fingers):

    if fingers == [1, 0, 0, 0]:
        return "DRAW"

    elif fingers == [1, 1, 0, 0]:
        return "PAUSE"

    elif fingers == [0, 0, 0, 0]:
        return "FIST"

    elif fingers == [1, 1, 1, 1]:
        return "OPEN PALM"

    else:
        return "UNKNOWN"
