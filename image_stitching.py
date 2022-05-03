""" image_stitching.py"""
"""
    定义了图像拼接类。要求图像文件与所在文件夹同名，且按拼接顺序排号。
"""
import math
import cv2
import easygui
import os.path
import numpy as np
from config import stitching_config as config

# 图像拼接类
class image_stitching:
    def __init__(self, file_name):
        # 初始化处理，加载文件
        self.config = config
        # 本次拼接的文件夹名称
        self.file_name = file_name
        # 待拼接图像列表
        self.images = []

        # 匹配出图片的格式，再把图片全都打开并保存在self.images里
        for format in self.config.image_format_list:
            if os.path.exists(f'./src/image/{self.file_name}/{self.file_name}_1.{format}'):
                self.work_path = f'./src/image/{self.file_name}'
                self.file_format = format

                # 从存储路径处加载全部图片
                i = 1
                while os.path.isfile(f'{self.work_path}/{self.file_name}_{i}.{self.file_format}'):
                    image = cv2.imread(f'{self.work_path}/{self.file_name}_{i}.{self.file_format}')
                    image = self.compress_picture(image)
                    self.images.append(image)
                    i += 1

                # 创建保存结果图和连线图的文件夹
                self.stitch_result_path = f'{self.work_path}/result'
                if not os.path.exists(self.stitch_result_path):
                    os.makedirs(self.stitch_result_path)

                self.image_matched_path = f'{self.work_path}/image_matched'
                if not os.path.exists(self.image_matched_path):
                    os.makedirs(self.image_matched_path)
                break

# public:
    # 图像拼接类外部接口函数
    def stitch(self):
        if self.config.stitch_style=='successive':
            images = []
            n = len(self.images)
            if self.config.stitch_order=='Sequence':
                images = self.images
                if (self.config.cylindrical_projection_run):
                    images = [self.cylindrical_projection(i) for i in images]
            else:
                if (n % 2 == 1):
                    pair = int(n / 2)
                    center = int((n - 1) / 2)
                    images.append(self.images[center])
                    for i in range(1, pair + 1):
                        images.append(self.images[center - i])
                        images.append(self.images[center + i])
                else:
                    pair = int(n / 2 - 1)
                    center = int((n - 1) / 2)
                    images.append(self.images[center])
                    for i in range(1, pair + 1):
                        images.append(self.images[center - i])
                        images.append(self.images[center + i])
                    images.append(self.images[n - 1])
                if (self.config.cylindrical_projection_run):
                    images = [self.cylindrical_projection(i) for i in images]


            if len(images) >= 2:
                image_fixed = images[0]
                i = 0
                for image_variable in images[1:]:
                    i += 1
                    # 计算出两张图片的关键点坐标列表与关键点描述子列表
                    key_points_variable, descriptors_variable = self.surf(image_variable)
                    key_points_fixed, descriptors_fixed = self.surf(image_fixed)

                    Homography_matrix, effective_key_points, matched_key_points = self.matchKeyPoints(image_variable,
                                                                                                      key_points_variable,
                                                                                                      key_points_fixed,
                                                                                                      descriptors_variable,
                                                                                                      descriptors_fixed)

                    # 如果继续拼接,需要匹配成功的关键点至少占所有有效关键点的key_points_ratio倍或数量至少超过要key_points_number个
                    key_points_ratio = 0.05 * config.key_points_judge_run
                    key_points_number = 20 * config.key_points_judge_run
                    stitch_threshold = min(key_points_ratio*len(key_points_variable), key_points_number)
                    if stitch_threshold < len(matched_key_points[matched_key_points == 1]):
                        still_stitch = '拼接'
                    else:
                        still_stitch = easygui.buttonbox("特征点太少,拼接效果可能不好,是否继续拼接?:", choices=['拼接', '不拼接'])

                    if still_stitch == '拼接':
                        self.drawMatches(image_variable, image_fixed,
                                         key_points_variable, key_points_fixed,
                                         effective_key_points, matched_key_points, i)
                        image_fixed = self.stitch_images(image_variable, image_fixed, Homography_matrix, i)
                cv2.imwrite(f'{self.stitch_result_path}/result.{self.config.imwrite_format}', image_fixed)
                return image_fixed

            else:
                easygui.msgbox("没有找到可拼接的图片")
                exit(0)
                return np.array([])
        else:
            if(len(self.images) < 2):
                easygui.msgbox("没有找到可拼接的图片")
                exit(0)
                return np.array([])

            images = self.images
            if (self.config.cylindrical_projection_run):
                images = [self.cylindrical_projection(i) for i in images]
            cnt = 0
            while(not len(images)==1):
                current = []
                pairs = math.floor(len(images)/2)
                single = ((len(images) % 2) == 1)
                for i in range(pairs):
                    cnt+=1
                    image_variable = images[2 * i + 1]
                    image_fixed = images[2 * i]
                    key_points_variable, descriptors_variable = self.surf(image_variable)
                    key_points_fixed, descriptors_fixed = self.surf(image_fixed)

                    Homography_matrix, effective_key_points, matched_key_points = self.matchKeyPoints(image_variable,
                                                                                                      key_points_variable,
                                                                                                      key_points_fixed,
                                                                                                      descriptors_variable,
                                                                                                      descriptors_fixed)

                    # 如果继续拼接,需要匹配成功的关键点至少占所有有效关键点的key_points_ratio倍或数量至少超过要key_points_number个
                    key_points_ratio = 0.05 * config.key_points_judge_run
                    key_points_number = 20 * config.key_points_judge_run
                    stitch_threshold = min(key_points_ratio * len(key_points_variable), key_points_number)
                    if stitch_threshold < len(matched_key_points[matched_key_points == 1]):
                        still_stitch = '拼接'
                    else:
                        still_stitch = easygui.buttonbox("特征点太少,拼接效果可能不好,是否继续拼接?:", choices=['拼接', '不拼接'])

                    if still_stitch == '拼接':
                        self.drawMatches(image_variable, image_fixed,
                                         key_points_variable, key_points_fixed,
                                         effective_key_points, matched_key_points, cnt)
                        image_fixed = self.stitch_images(image_variable, image_fixed, Homography_matrix, cnt)
                    current.append(image_fixed)
                if single:
                    current.append(images[-1])
                images = current
            cv2.imwrite(f'{self.stitch_result_path}/result.{self.config.imwrite_format}', images[0])
            return images[0]


    # private:
    # 压缩图像
    def compress_picture(self, image):
        """把图像压缩到self.setting.image_size,如果是reel模式，那就只限制高度"""
        if self.config.cut_style == 'reel':
            # compression_ratio = image.shape[0] / np.sqrt(self.config.image_size)
            compression_ratio = np.sqrt(image.shape[0] * image.shape[1] / self.config.image_size)
        else:
            compression_ratio = np.sqrt(image.shape[0] * image.shape[1] / self.config.image_size)
        if compression_ratio > 1:
            image = cv2.resize(image, (int(image.shape[1]/compression_ratio), int(image.shape[0]/compression_ratio)))

        return image

    # 柱面变换
    def cylindrical_projection(self, image):
        result = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        depth = image.shape[2]

        center_x = int(width / 2)
        center_y = int(height / 2)
        alpha = math.pi / 4
        f = width / (2 * math.tan(alpha / 2))
        for x in range(width):
            for y in range(height):
                theta = math.atan((x - center_x) / f)
                point_x = f * math.tan((x - center_x) / f) + center_x
                point_y = (y - center_y) / math.cos(theta) + center_y

                for c in range(depth):
                    if point_x >= width-1 or point_x < 0 or point_y >= height-1 or point_y < 0:
                        result[y, x, c] = 0
                    else:
                        if self.config.bilinear_interpolation:
                            u = point_x - int(point_x)
                            v = point_x - int(point_x)
                            s1 = image[int(point_y), int(point_x), c]
                            s2 = image[int(point_y), int(point_x) + 1, c]
                            s3 = image[int(point_y) + 1, int(point_x), c]
                            s4 = image[int(point_y) + 1, int(point_x) + 1, c]
                            result[y, x, c] = (1 - u)*(1 - v)*s1 + (1 - u)*v*s2 + u*(1 - v)*s3 + u*v*s4
                        else:
                            result[y, x, c] = image[int(point_y), int(point_x), c]
        return result

    # 计算image图像的所有关键点的坐标及其描述子
    def surf(self, image):
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

    # 特征点匹配
    def matchKeyPoints(self, image_variable, key_points_variable, key_points_fixed,
                       descriptors_variable, descriptors_fixed,
                       ratio=0.75, ransac_thresh=4.0):
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
        key_points_fixed = np.float32([key_points_fixed[m[1]] + [self.config.k*np.size(image_variable, 1), self.config.k*np.size(image_variable, 0)] for m in effective_key_points])

        # RANSAC算法计算单应矩阵，得到匹配成功的有效关键点对
        Homography_matrix, matched_key_points = cv2.findHomography(key_points_variable, key_points_fixed, cv2.RANSAC, ransac_thresh)

        return Homography_matrix, effective_key_points, matched_key_points

    def drawMatches(self, image_variable, image_fixed, key_points_variable, key_points_fixed,
                    effective_key_points, matched_key_points, i):
        # 标出两个图片的匹配特征点连线,匹配成功的连红线,不成功的连绿线
        if self.config.image_matched_run:
            # 由于矩阵是先行号后列号,与x,y坐标系相反,所以用height与weight描述图片
            (h_variable, w_variable) = image_variable.shape[0:2]
            (h_fixed, w_fixed) = image_fixed.shape[0:2]

            # 按照固定图片在左,变化图片在右的顺序,并排居中放置两张图片
            image_matched_h = max(h_variable, h_fixed)
            image_matched_w = w_variable + w_fixed
            image_matched = np.zeros((image_matched_h, image_matched_w, 3), 'uint8')

            # 左右两张图片的左上角位置,即原点位置,保存在o_variable与o_fixed中
            o_variable = [(image_matched_h - h_variable)//2, w_fixed]
            o_fixed = [(image_matched_h - h_fixed)//2, 0]
            image_matched[o_fixed[0]:o_fixed[0]+h_fixed, o_fixed[1]:o_fixed[1]+w_fixed] = image_fixed
            image_matched[o_variable[0]:o_variable[0]+h_variable, o_variable[1]:o_variable[1]+w_variable] = image_variable

            # 计算匹配上的特征点在目前画布上的位置,在可以继续画时,匹配成功的标红线,不成功的标绿线
            r_already = 0
            g_already = 0
            for ((queryIdx, trainIdx), s) in zip(effective_key_points, matched_key_points):
                pt1 = (int(key_points_fixed[trainIdx][0]) + o_fixed[1],
                       int(key_points_fixed[trainIdx][1]) + o_fixed[0])
                pt2 = (int(key_points_variable[queryIdx][0]) + o_variable[1],
                       int(key_points_variable[queryIdx][1]) + o_variable[0])
                if s == 1 and r_already < self.config.image_matched_r:
                    cv2.line(image_matched, pt1, pt2, (0, 0, 255))
                    r_already += 1
                elif s == 0 and g_already < self.config.image_matched_g:
                    cv2.line(image_matched, pt1, pt2, (0, 255, 0))
                    g_already += 1

            cv2.imwrite(f'{self.image_matched_path}/image_matched_{i}.{self.config.imwrite_format}', image_matched)

    # 拼接函数，把可变图像映射到画布中，然后进行图像融合
    def stitch_images(self, image_variable, image_fixed, Homography_matrix, i):
        # 画布尺寸canvas_size,把固定图像放在中心,上下左右留出self.config.k倍可变图像的冗余
        canvas_size = (int(image_variable.shape[0]*2*self.config.k + image_fixed.shape[0]),
                           int(image_variable.shape[1]*2*self.config.k + image_fixed.shape[1]))

        # 可变图像画布
        # 可变图像通过单应矩阵映射到variable_canvas
        variable_canvas = cv2.warpPerspective(image_variable, Homography_matrix, (canvas_size[1], canvas_size[0]))
        # 二值化并腐蚀边缘
        bi_variable_canvas = variable_canvas.copy()
        bi_variable_canvas[bi_variable_canvas != 0] = 1
        bi_variable_canvas[np.any(bi_variable_canvas[..., :] == 1, axis=2)] = 1
        bi_variable_canvas = cv2.erode(bi_variable_canvas, None, iterations=self.config.erode)
        variable_canvas = variable_canvas * bi_variable_canvas

        # 固定图像画布
        fixed_canvas = np.zeros((canvas_size[0], canvas_size[1], 3))
        fixed_canvas[int(image_variable.shape[0]*self.config.k):int(image_variable.shape[0]*self.config.k+image_fixed.shape[0]), int(image_variable.shape[1]*self.config.k):int(image_variable.shape[1]*self.config.k+image_fixed.shape[1])] = image_fixed
        bi_fixed_canvas = fixed_canvas.copy()
        bi_fixed_canvas[bi_fixed_canvas != 0] = 1
        bi_fixed_canvas[np.any(bi_fixed_canvas[..., :] == 1, axis=2)] = 1
        # 重叠部分的二值图像
        bi_overlap_canvas = bi_variable_canvas * bi_fixed_canvas

        if self.config.p_trans_style=='average':
            p_variable = 0.5 * bi_overlap_canvas
            p_fixed = 0.5 * bi_overlap_canvas
        else:
            # 两个0~1递增矩阵,用来计算融合系数
            if self.config.trans_style=='edge_to_center':
                trans_variable = self.edge_to_center(image_variable.shape[0], image_variable.shape[1])
                trans_fixed = self.edge_to_center(image_fixed.shape[0], image_fixed.shape[1])
            else:
                trans_variable = self.side_to_edge(image_variable.shape[0], image_variable.shape[1])
                trans_fixed = self.side_to_edge(image_fixed.shape[0], image_fixed.shape[1])
            trans_variable_canvas = cv2.warpPerspective(trans_variable, Homography_matrix, (canvas_size[1], canvas_size[0]))
            # trans_variable_canvas = trans_variable_canvas * bi_variable_canvas
            trans_fixed_canvas = np.ones((canvas_size[0], canvas_size[1], 3))
            trans_fixed_canvas[int(image_variable.shape[0]*self.config.k):int(image_variable.shape[0]*self.config.k+image_fixed.shape[0]), int(image_variable.shape[1]*self.config.k):int(image_variable.shape[1]*self.config.k+image_fixed.shape[1])] = trans_fixed

            # 计算两张图片重叠部分的显示比例
            p_variable = trans_variable_canvas*bi_overlap_canvas / (trans_variable_canvas + trans_fixed_canvas)
            p_fixed = trans_fixed_canvas*bi_overlap_canvas / (trans_variable_canvas + trans_fixed_canvas)
            if self.config.p_trans_show:
                self.show_picture(p_fixed*bi_overlap_canvas, 'Fixed picture display ratio')
                self.show_picture(p_variable*bi_overlap_canvas, 'Variable picture display ratio')

        stitching_result = np.zeros((canvas_size[0], canvas_size[1], 3))
        if self.config.image_fusion_run:
            if self.config.multi_band_blend:
                if self.config.pyr_levels == 'max':
                    levels = math.floor(math.log(min(canvas_size), 2))
                else:
                    levels = self.config.pyr_levels
                mask_fixed = p_fixed + (bi_fixed_canvas - bi_overlap_canvas)
                mask_variable = p_variable + (bi_variable_canvas - bi_overlap_canvas)

                A = fixed_canvas.astype(np.float64) / 255.0
                B = variable_canvas.astype(np.float64) / 255.0

                mask_A = self.gaussianPyramid(mask_fixed, levels)
                mask_B = self.gaussianPyramid(mask_variable, levels)

                G_A = self.gaussianPyramid(A, levels)
                L_A = self.laplacianPyramid(G_A)

                G_B = self.gaussianPyramid(B, levels)
                L_B = self.laplacianPyramid(G_B)

                bPy = self.blendPyramid(L_A, L_B, mask_A, mask_B)

                stitching_result = self.collapsePyramid(bPy)

            else:
                stitching_result += (1 - bi_overlap_canvas) * (fixed_canvas + variable_canvas)
                stitching_result += bi_overlap_canvas * (p_fixed * fixed_canvas + p_variable * variable_canvas)
        else:
            stitching_result += (1 - bi_overlap_canvas) * (fixed_canvas + variable_canvas)
            stitching_result += bi_overlap_canvas * variable_canvas

        if self.config.stitch_canvas_show:
            self.show_picture(fixed_canvas, 'fixed_canvas')
            self.show_picture(variable_canvas, 'variable_canvas')
            self.show_picture(stitching_result, 'stitching_result')

        if self.config.cut_style == 'reel':
            if self.config.cylindrical_projection_run:
                idx = np.argwhere(np.all(stitching_result[..., :] <= 20, axis=1))
                stitching_result = np.delete(stitching_result, idx, axis=0)
                idx = np.argwhere(np.all(stitching_result[..., :] <= 20, axis=0))
                stitching_result = np.delete(stitching_result, idx, axis=1)

            else:
                stitching_result = stitching_result[int(image_variable.shape[0] * self.config.k):
                                                    int(image_variable.shape[0] * self.config.k) + image_fixed.shape[0]]
                bi_stitching_result = bi_fixed_canvas + bi_variable_canvas - bi_overlap_canvas
                bi_stitching_result = bi_stitching_result[int(image_variable.shape[0] * self.config.k):
                                                          int(image_variable.shape[0] * self.config.k) + image_fixed.shape[0]]
                idx = np.argwhere(np.any(bi_stitching_result[..., :] == 0, axis=0))
                stitching_result = np.delete(stitching_result, idx, axis=1)
        else:
            idx = np.argwhere(np.all(stitching_result[..., :] <= 20, axis=1))
            stitching_result = np.delete(stitching_result, idx, axis=0)
            idx = np.argwhere(np.all(stitching_result[..., :] <= 20, axis=0))
            stitching_result = np.delete(stitching_result, idx, axis=1)

        stitching_result = self.compress_picture(stitching_result)
        self.show_picture(stitching_result, 'stitching result')
        stitching_result = stitching_result.astype(np.uint8)
        cv2.imwrite(f'{self.stitch_result_path}/temp_{i}.{self.config.imwrite_format}', stitching_result)
        return stitching_result

    def show_picture(self, image, title='temp'):
        if self.config.stitch_temp_image_show:
            image_copy = image.copy()
            image_copy = self.compress_picture(image_copy)

            # 如果是二值图像就放大成全白的显示，否则变成整型显示
            if int(np.max(image_copy)) <= 2:
                image_copy = image_copy*255
                image_copy = image_copy.astype(np.uint8)
                cv2.imshow(title, image_copy)
            else:
                image_copy = image_copy.astype(np.uint8)
                cv2.imshow(title, image_copy)

            cv2.waitKey(0)
            cv2.destroyWindow(title)


    def edge_to_center(self, m, n):
        # 生成重叠部分到中心的0~1递增矩阵
        trans = np.ones((m, n))
        i = 2
        while 2 * i <= min(m, n):
            trans[i:m-i, i:n-i] += 1
            i += 1
        trans = trans / np.max(trans)
        trans = np.power(trans, self.config.power)

        trans = np.array([trans, trans, trans])
        trans = trans.transpose((1, 2, 0))
        return trans

    def side_to_edge(self, m, n):
        # 生成边界到重叠部分的0~1递增矩阵
        trans = np.zeros((m, n))
        for y in range(m):
            for x in range(n):
                trans[y, x] = min([x + 1, y + 1, n - x, m - y])
        trans = trans / np.max(trans)
        trans = np.power(trans, self.config.power)

        trans = np.array([trans, trans, trans])
        trans = trans.transpose((1, 2, 0))
        return trans

    def gaussianPyramid(self, image, levels):
        result = []
        image = np.array(image, dtype=np.float64)
        result.append(image)
        current = image
        for i in range(levels-1):
            _pyrDown = cv2.pyrDown(current)
            result.append(_pyrDown)
            current = _pyrDown
        return result

    def laplacianPyramid(self, gaussianPyramid):
        levels = len(gaussianPyramid)
        result = []
        result.append(gaussianPyramid[levels-1])
        i = levels-2
        while(i>=0):
            _pyrUp = None
            _pyrUp = cv2.pyrUp(gaussianPyramid[i+1], _pyrUp, gaussianPyramid[i].shape[-2::-1])
            current = gaussianPyramid[i] - _pyrUp
            result.append(current)
            i = i - 1
        return result

    def blendPyramid(self, pyrA, pyrB, maskA, maskB):
        levels = len(pyrA)
        result = []
        for i in range(levels):
            result.append(np.multiply(pyrA[i], maskA[levels - 1 - i]) + np.multiply(pyrB[i], maskB[levels - 1 - i]))
        return result

    def collapsePyramid(self, blendPyramid):
        levels = len(blendPyramid)
        current = blendPyramid[0]
        for i in range(1, levels):
            current = cv2.pyrUp(current, current, blendPyramid[i].shape[-2::-1])
            current += blendPyramid[i]
        result = current

        mn = result.min()
        mx = result.max()
        mx -= mn
        result = (result - mn) / mx
        result = abs(result * 255.0).astype('uint8')
        return result