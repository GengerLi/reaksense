import cv2
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        print(f"HSV at ({x},{y}): {pixel}")

img = cv2.imread("jiaodai3.png") #传入hsv图像

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
