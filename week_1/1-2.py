import cv2
import numpy as np

drawing = False
color = (255, 0, 0)
brush_size = 5

def paint(event, x, y, flags, param):
    global drawing, color

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0)
        cv2.circle(img, (x, y), brush_size, color, -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255)
        cv2.circle(img, (x, y), brush_size, color, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), brush_size, color, -1)
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing = False

img = np.ones((600, 800, 3), dtype=np.uint8) * 255
cv2.namedWindow('Paint')
cv2.setMouseCallback('Paint', paint)

while True:
    cv2.imshow('Paint', img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        brush_size = min(15, brush_size + 1)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

cv2.destroyAllWindows()