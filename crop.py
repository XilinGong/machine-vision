import cv2
import numpy as np

result = cv2.imread("hard_result.jpg")
stitched = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(stitched.astype("uint8"), cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# 轮廓最小正矩形
mask = np.zeros(thresh.shape, dtype="uint8")
(x, y, w, h) = cv2.boundingRect(cnts)  # 取出list中的轮廓二值图，类型为numpy.ndarray
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

# 腐蚀处理，直到minRect的像素值都为0
minRect = mask.copy()
sub = mask.copy()
while cv2.countNonZero(sub) > 0:
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)

h = np.where(minRect==255)[0]
w = np.where(minRect==255)[1]

result = result[h[0]:h[-1], w[0]:w[-1], :]
cv2.imwrite("s.jpg", result)
print("right")