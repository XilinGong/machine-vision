""" objdect.py"""
"""
    定义了目标检测类。可以直接传入ndarray图像，或给出图片路径。
"""
import cv2
import matplotlib.pyplot as plt
from selectivesearch import selective_search
from resnet_for_image_classify import predict
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from config import detect_config as config


class Dector:
    def __init__(self, image_path=None, image=None):
        self.config = config

        if config.modelsize == "resnet50":
            self.model = ResNet50(weights='imagenet')
            pass
        elif config.modelsize == "resnet152":
            self.model = ResNet152(weights='imagenet')
            pass
        else:
            raise NotImplementedError("please select an available method, resnet50 or resnet152")

        if image_path is not None:
            img = cv2.imread(image_path)
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pass
        else:
            self.image = image
            self.image_save = image
        self.w, self.h = self.image.shape[:2]
        self.w_save = self.h_save = self.image_save.shape[:2]

    def __call__(self):
        if self.config.resize:
            self.resize()
            pass

        rects = self.selective_search()
        self.show_rect(rects)
        obj_boxes, all_boxes = self.predict_class(rects)

        final_obj_boxes = dict()
        final_all_boxes = dict()
        self.show_picture(obj_boxes, 0)

        for key in obj_boxes.keys():
            final_obj_boxes[key] = self.nms(obj_boxes[key], self.config.nms_method)
            pass
        print(final_obj_boxes)
        self.show_picture(final_obj_boxes, 1)

        # for key in all_boxes.keys():
        #     final_all_boxes[key] = self.nms(all_boxes[key], self.config.nms_method)
        # print(final_all_boxes)
        # self.show_picture(final_all_boxes)

    def resize(self):
        self.w = int(self.w/self.config.resize_rate)
        self.h = int(self.h/self.config.resize_rate)
        self.image = cv2.resize(self.image, (self.h, self.w))

    def selective_search(self):
        if self.config.find_rect_method == "ss":
            '''使用ss方法'''
            enum, regions = selective_search(self.image, scale=500, sigma=0.8, min_size=200)
            # 创建储存候选框的容器
            candidates = set()
            # 遍历产生的候选框
            for r in regions:
                # 如果当前候选框已经存在在candidates内，忽略
                if r['rect'] in candidates:
                    continue
                # 删除过于小的框
                _, _, w, h = r['rect']
                if w * h < config.mintargetsize:
                    continue
                # 删除畸变过于严重的框
                if w / h > 3 or h / w > 3:
                    continue
                # 将满足条件的候选框加入candidates
                candidates.add(r['rect'])
                pass
            pass

        elif self.config.find_rect_method == "gradient_mv":
            '''使用gradient-moving-window方法'''
            pass
        else:
            raise NotImplementedError("please select an available method,ss or gradient_mv")
        return candidates

    def predict_class(self, propal_rects):
        print("{} targets being predicted by ".format(len(propal_rects))+config.modelsize)

        preds = [] # 存放成功预测的框
        finded_obj_class = [] # 存放最终的框
        for _, rect in enumerate(propal_rects):
            x, y, w, h = rect # 得到四元组 w，h表宽和高
            # 调用神经网络进行预测 返回全部预测结果
            pred = predict(self.model,
                           self.config.target_size,
                           self.config.top_n,
                           self.image[y:y+h, x:x+w])
            # 除去小于最低阈值的框 同时统计都检测出来了哪些类
            for i in range(self.config.top_n):
                if pred[i][2] > self.config.threshold_for_resnet:
                    preds.append((pred[i], x, y, w, h))
                    if pred[i][1] not in finded_obj_class:
                        finded_obj_class.append(pred[i][1])
                        pass
                    pass
                pass
            pass
        # 打印可能存在的类 便于找到需要的
        print(finded_obj_class)
        # 建立一个字典来储存不同类别物体的预测框
        finded_bbox_dict = dict()
        for obj in finded_obj_class:
            finded_bbox_dict[obj] = []
            for (pred, x, y, w, h) in preds:
                if pred[1] == obj:
                    finded_bbox_dict[obj].append((pred[2], x, y, w, h))  # {classname} [(score,xywh)]
                    pass
                pass
            pass
        # 建立一个字典存储想要找到的物体的预测框
        object_bbox_dict = dict()
        # 应当待寻找到类别中
        for obj in self.config.target_obj:
            object_bbox_dict[obj] = []
            for (pred, x, y, w, h) in preds:
                if pred[1] == obj:
                    object_bbox_dict[obj].append((pred[2], x, y, w, h))  # {classname} [(score,xywh)]
                    pass
                pass
            pass
        return object_bbox_dict, finded_bbox_dict

    def show_rect(self, rects):
        plt.figure()
        plt.imshow(self.image)
        ax = plt.gca()
        for x, y, w, h in rects:
            rect = plt.Rectangle((x, y), w, h, color="red", fill=False, linewidth=1)
            ax.add_patch(rect)
            pass
        plt.savefig("result/rects" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
        plt.clf()

    def show_picture(self, bboxes, nmsed):   # dict
        plt.figure()
        plt.imshow(self.image_save)
        ax = plt.gca()
        for key in bboxes.keys():
            for bbox in bboxes[key]:
                score, x, y, w, h = bbox
                rect = plt.Rectangle((x * self.config.resize_rate,
                                      y * self.config.resize_rate),
                                     w * self.config.resize_rate,
                                     h * self.config.resize_rate,
                                     color="red", fill=False, linewidth=1)
                ax.add_patch(rect)
                ax.text(x * self.config.resize_rate,
                        y * self.config.resize_rate,
                        "{}".format(key)+'{:.2f}'.format(score),
                        bbox={'facecolor': 'blue', 'alpha': 0})
                pass
        if nmsed:
            plt.savefig("result/obj_nmsed" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
            pass
        else:
            plt.savefig("result/obj" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
        plt.clf()

    '''两种非极大值抑制方法'''
    def nms(self, bboxes, method):
        finalbox = []
        # nms
        if method == 'nms':
            while len(bboxes) > 0:
                bboxes.sort(key=lambda s: s[0], reverse=True)
                best = bboxes[0]
                finalbox.append(bboxes.pop(0))
                poplist = []
                for bbox in bboxes:
                    metric = self.iou((best[1], best[2], best[3], best[4]),
                                      (bbox[1], bbox[2], bbox[3], bbox[4]),
                                      self.config.iou_method)
                    if metric > self.config.threshold_for_iou:
                        poplist.append(bbox)
                        pass
                    pass
                while len(poplist) > 0:
                    bboxes.remove(poplist.pop())
                    pass
                pass
            pass
        # soft_nms
        elif method == 'soft_nms':
            while len(bboxes) > 0:
                bboxes.sort(key=lambda s: s[0], reverse=True)
                best = bboxes[0]
                finalbox.append(bboxes.pop(0))
                for bbox in bboxes:
                    metric = self.iou((best[1], best[2], best[3], best[4]),
                                      (bbox[1], bbox[2], bbox[3], bbox[4]),
                                      self.config.iou_method)
                    # 如果高于阈值，则进行得分衰减，衰减程度与metric大小有关
                    if metric > self.config.threshold_for_iou:
                        bbox[0] = bbox[0]*(1-metric)
                        pass
                    pass
                pass
            poplist = []
            for fb in finalbox:
                if fb[0] < config.threshold_for_resnet:
                    poplist.append(fb)
                    pass
                pass
            while len(poplist) > 0:
                finalbox.remove(poplist.pop())
                pass
        else:
            raise NotImplementedError("please select an available method, nms or soft_nms")
        return finalbox

    '''三种指标的计算'''
    def iou(self, bbox1, bbox2, method):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        in_h = min(y1 + h1, y2 + h2) - max(y1, y2)
        in_w = min(x1 + w1, x2 + w2) - max(x1, x2)

        if in_h <= 0 or in_w <= 0:
            inner = 0
            pass
        else:
            inner = in_h * in_w
        # iou
        if method == 'iou':
            metric = inner / (w1 * h1 + w2 * h2 - inner)
            pass
        # giou
        elif method == 'giou':
            out_h = max(y1 + h1, y2 + h2) - min(y1, y2)
            out_w = max(x1 + w1, x2 + w2) - min(x1, x2)
            outer = out_w * out_h
            metric = inner / (w1 * h1 + w2 * h2 - inner) - \
                     (outer - (w1 * h1 + w2 * h2 - inner)) / outer  # [-1,1]
            metric = (metric + 1) / 2.0  # change to [0,1]
            pass
        # diou
        elif method == 'diou':
            bbox1_center_x = x1 + w1 / 2
            bbox1_center_y = y1 + h1 / 2
            bbox2_center_x = x2 + w2 / 2
            bbox2_center_y = y2 + h2 / 2
            out_h = max(y1 + h1, y2 + h2) - min(y1, y2)
            out_w = max(x1 + w1, x2 + w2) - min(x1, x2)
            d_2 = pow(bbox2_center_x-bbox1_center_x, 2)+pow(bbox2_center_y-bbox1_center_y, 2)
            c_2 = pow(out_w, 2)+pow(out_h, 2)
            metric = inner / (w1 * h1 + w2 * h2 - inner) - d_2 / c_2    # [-1,1]
            metric = (metric + 1) / 2.0  # change to [0,1]
            pass
        else:
            raise NotImplementedError("please select an available method,iou giou or diou")
        return metric


if __name__ == "__main__":
    dector = Dector("src/image/hard/test1.jpg")
    dector()
