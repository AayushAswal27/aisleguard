import cv2

IMG_PATH = "outputs/calib_frame.jpg"
img = cv2.imread(IMG_PATH)
clicks = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Point {len(clicks)}: ({x}, {y})")
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, str(len(clicks)), (x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("calib", img)
        if len(clicks) == 4:
            print("\n4 points captured. Press any key to close.")

cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
cv2.imshow("calib", img)
cv2.setMouseCallback("calib", on_click)
print("Click 4 points in order: near-left, near-right, far-left, far-right")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nPIXEL_POINTS =", clicks)
