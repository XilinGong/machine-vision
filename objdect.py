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
        elif config.modelsize == "resnet152":
            self.model = ResNet152(weights='imagenet')

        if image_path is not None:
            img = cv2.imread(image_path)
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self.image = image
            self.image_save = image
        self.w, self.h = self.image.shape[:2]
        self.w_save = self.h_save = self.image_save.shape[:2]

    def __call__(self):
        if self.config.resize:
            self.resize()

        rects = self.selective_search()
        self.show_rect(rects)
        obj_boxes, all_boxes = self.predict_class(rects)

        final_obj_boxes = dict()
        final_all_boxes = dict()

        for key in obj_boxes.keys():
            final_obj_boxes[key] = self.nms(obj_boxes[key], self.config.nms_method)
        print(final_obj_boxes)
        self.show_picture(final_obj_boxes)

        # for key in all_boxes.keys():
        #     final_all_boxes[key] = self.nms(all_boxes[key], self.config.nms_method)
        # print(final_all_boxes)
        # self.show_picture(final_all_boxes)

    def resize(self):
        self.w = int(self.w/self.config.resize_rate)
        self.h = int(self.h/self.config.resize_rate)
        self.image = cv2.resize(self.image, (self.h, self.w))

    def selective_search(self):
        enum, regions = selective_search(self.image, scale=200, sigma=0.8, min_size=500)
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
            # 将满足条件的候选框加入candidates
            candidates.add(r['rect'])
        return candidates

    def predict_class(self, propal_rects):
        print("{} targets being predicted by ".format(len(propal_rects))+config.modelsize)

        preds = []
        finded_obj_class = []
        for _, rect in enumerate(propal_rects):
            x, y, h, w = rect
            pred = predict(self.model,
                           self.config.target_size,
                           self.config.top_n,
                           self.image[y:y+h, x:x+w])

            for i in range(self.config.top_n):
                if pred[i][2] > self.config.threshold_for_resnet:
                    preds.append((pred[i], x, y, w, h))
                    if pred[i][1] not in finded_obj_class:
                        finded_obj_class.append(pred[0][1])

        print("{} targets founded by ".format(len(preds))+self.config.modelsize)

        # 建立一个字典来储存不同类别物体的预测框
        finded_bbox_dict = dict()
        for obj in finded_obj_class:
            finded_bbox_dict[obj] = []
            for (pred, x, y, w, h) in preds:  # pred(nqohxn classname  score),x,y,w,h
                if pred[1] == obj:
                    finded_bbox_dict[obj].append((pred[2], x, y, w, h))  # {classname} [(score,xywh)]

        # self.show_picture(finded_bbox_dict)

        object_bbox_dict = dict()
        for obj in self.config.target_obj:
            object_bbox_dict[obj] = []
            for (pred, x, y, w, h) in preds:    # pred(nqohxn classname  score),x,y,w,h
                if pred[1] == obj:
                    object_bbox_dict[obj].append((pred[2], x, y, w, h))  # {classname} [(score,xywh)]

        print(object_bbox_dict)
        return object_bbox_dict, finded_bbox_dict

    def show_rect(self, rects):
        plt.figure()
        plt.imshow(self.image)
        ax = plt.gca()
        for x, y, w, h in rects:
            rect = plt.Rectangle((x, y), w, h, color="red", fill=False, linewidth=1)
            ax.add_patch(rect)
        plt.savefig("rects" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
        plt.show()
        plt.clf()

    def show_picture(self, bboxes):   # dict
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

        plt.savefig("object_detection" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
        plt.show()
        plt.clf()

    def nms(self, bboxes, method):
        finalbox = []

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
                while len(poplist) > 0:
                    bboxes.remove(poplist.pop())
        return finalbox

    def iou(self, bbox1, bbox2, method):
        metric = 0

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        in_h = min(y1 + h1, y2 + h2) - max(y1, y2)
        in_w = min(x1 + w1, x2 + w2) - max(x1, x2)

        if in_h <= 0 or in_w <= 0:
            inner = 0
        else:
            inner = in_h * in_w

        if method == 'iou':
            metric = inner / (w1 * h1 + w2 * h2 - inner)

        elif method == 'giou':
            out_h = max(y1 + h1, y2 + h2) - min(y1, y2)
            out_w = max(x1 + w1, x2 + w2) - min(x1, x2)
            outer = out_w * out_h
            metric = inner / (w1 * h1 + w2 * h2 - inner) - \
                     (outer - (w1 * h1 + w2 * h2 - inner)) / outer  # [-1,1]
            metric = (metric + 1) / 2.0  # change to [0,1]

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

        return metric


if __name__ == "__main__":
    dector = Dector("src/image/bj2.png")
    dector()
