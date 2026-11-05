import cv2


def draw_line(canvas, start_point, end_point, color, thickness=5):
    """
    Draw a line between two points on the canvas.
    """
    if start_point is None or end_point is None:
        return

    cv2.line(canvas, start_point, end_point, color, thickness)


def clear_canvas(canvas):
    """
    Clear the entire drawing canvas.
    """
    canvas[:] = 0
    return canvas