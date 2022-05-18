import cv2
import numpy as np
from config import stitching_config as config


def compress_picture(image):
    compression_ratio = np.sqrt(image.shape[0] * image.shape[1] / 400000)
    if compression_ratio > 1:
        image = cv2.resize(image, (int(image.shape[1] / compression_ratio), int(image.shape[0] / compression_ratio)))

    return image

def surf(image):
    """
    Output:
        key_points: n*2维矩阵,代表每个关键点的x,y坐标
        descriptors: n*64维矩阵,代表每个关键点的描述子
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    key_points, descriptors = detector.detectAndCompute(gray, None)

    # 转换为np格式
    key_points = np.float32([kp.pt for kp in key_points])

    return key_points, descriptors

def matchKeyPoints(image_variable, key_points_variable, key_points_fixed, descriptors_variable, descriptors_fixed, ratio=0.75, ransac_thresh=4.0):
    """
    Input:
        ratio: 是最近邻匹配的推荐阈值
        ransac_thresh: 是随机取样一致性的推荐阈值
    Output:
        Homography_matrix: 可变图像到固定图像的单应矩阵
        effective_key_points: 有效的关键点对
        matched_key_points: 匹配成功的关键点对
    """
    # 构建暴力匹配器，使用knn算法每次从描述子向量集中找到最近的两个向量
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(descriptors_variable, descriptors_fixed, 2)

    # 得到有效的关键点
    effective_key_points = []
    for m in rawMatches:
        # 因为关键点都是一对一的，所以最近的向量应该比次近的向量的ratio倍要小
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            effective_key_points.append((m[0].queryIdx, m[0].trainIdx))

    # 建立有效关键点的映射关系,由于后期拼接的时候上下左右都留有空隙,所以注意可变图像的偏移,需要把它们的偏移位置调好
    key_points_variable = np.float32([key_points_variable[m[0]] for m in effective_key_points])
    key_points_fixed = np.float32([key_points_fixed[m[1]] + [config.k*np.size(image_variable, 1), config.k*np.size(image_variable, 0)] for m in effective_key_points])

    # RANSAC算法计算单应矩阵，得到匹配成功的有效关键点对
    Homography_matrix, matched_key_points = cv2.findHomography(key_points_variable, key_points_fixed, cv2.RANSAC, ransac_thresh)

    return Homography_matrix, effective_key_points, matched_key_points



img1 = cv2.imread('./src/image/image_stitching/test/test_1.jpg')
img2 = cv2.imread('./src/image/image_stitching/test/test_2.jpg')
image_fixed = compress_picture(img1)
image_variable = compress_picture(img2)
key_points_variable, descriptors_variable = surf(image_variable)
key_points_fixed, descriptors_fixed = surf(image_fixed)
Homography_matrix, effective_key_points, matched_key_points = matchKeyPoints(image_variable,
                                                                             key_points_variable, key_points_fixed,
                                                                             descriptors_variable, descriptors_fixed)
canvas_size = (int(image_variable.shape[0]*2*config.k + image_fixed.shape[0]),
               int(image_variable.shape[1]*2*config.k + image_fixed.shape[1]))
# 可变图像画布
# 可变图像通过单应矩阵映射到variable_canvas
variable_canvas = cv2.warpPerspective(image_variable, Homography_matrix, (canvas_size[1], canvas_size[0]))
# 二值化并腐蚀边缘
bi_variable_canvas = variable_canvas.copy()
bi_variable_canvas[bi_variable_canvas != 0] = 1
bi_variable_canvas[np.any(bi_variable_canvas[..., :] == 1, axis=2)] = 1
bi_variable_canvas = cv2.erode(bi_variable_canvas, None, iterations=config.erode)
variable_canvas = variable_canvas * bi_variable_canvas

# 固定图像画布
fixed_canvas = np.zeros((canvas_size[0], canvas_size[1], 3))
fixed_canvas[int(image_variable.shape[0]*config.k):int(image_variable.shape[0]*config.k+image_fixed.shape[0]), int(image_variable.shape[1]*config.k):int(image_variable.shape[1]*config.k+image_fixed.shape[1])] = image_fixed
bi_fixed_canvas = fixed_canvas.copy()
bi_fixed_canvas[bi_fixed_canvas != 0] = 1
bi_fixed_canvas[np.any(bi_fixed_canvas[..., :] == 1, axis=2)] = 1
# 重叠部分的二值图像
bi_overlap_canvas = bi_variable_canvas * bi_fixed_canvas

v1 = 0
v2 = 0
for i in range(canvas_size[0]):
    for j in range(canvas_size[1]):
        v1 += 0.59 * fixed_canvas[i,j,0]*bi_overlap_canvas[i,j,0] + 0.11*fixed_canvas[i,j,1]*bi_overlap_canvas[i,j,1] + 0.3*fixed_canvas[i,j,2]*bi_overlap_canvas[i,j,2]
        v2 += 0.59 * variable_canvas[i, j, 0] * bi_overlap_canvas[i, j, 0] + 0.11 * variable_canvas[i, j, 1] * bi_overlap_canvas[i, j, 1] + 0.3 * variable_canvas[i, j, 2] * bi_overlap_canvas[i, j, 2]

print(v1)
print(v2)
v = v1/v2
print(v)

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        img2[i,j,:] = img2[i,j,:]*v

cv2.imwrite('./src/image/image_stitching/test/testing_hard_1.jpg', img2)