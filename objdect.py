import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import time
from resnet_for_image_classify import predict
from tensorflow.keras.applications.resnet50 import ResNet50
from config import detect_config as config
import cv2
import numpy as np
import os
import easygui
import datetime


class Dector:
    def __init__(self, image_path=None,image=None):
        self.model = ResNet50(weights='imagenet')
        self.config = config
        if image_path!=None:
            self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.image_save = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        else:
            self.image = image
            self.image_save = image
        self.w, self.h = self.image.shape[:2]
        self.w_save = self.h_save = self.image_save.shape[:2]

    def __call__(self):
        if self.config.resize:
            self.resize()
        rects = self.selective_search()
        #self.show_rect(rects)
        obj_boxes, all_boxes= self.predict_class(rects)
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
        pass

    def resize(self):
        self.w = int(self.w/self.config.resize_rate)
        self.h = int(self.h/self.config.resize_rate)
        self.image = cv2.resize(self.image, (self.h, self.w))

    def selective_search(self):
        # 创建选择性搜索分割对象
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        # 设置输入图像，我们将运行分割
        ss.setBaseImage(self.image)
        # 快速但低召回选择性搜索方法
        if self.config.ss_methods[self.config.ss_method] == "fast":
            ss.switchToSelectiveSearchFast()
        # 高召回但慢选择性搜索方法
        elif self.config.ss_methods[self.config.ss_method] == "quality":
            ss.switchToSelectiveSearchQuality()
        # 运行选择性搜索分割输入图像
        rects = ss.process()
        return rects

    def predict_class(self, propal_rects):
        preds=[]
        finded_obj_class = []
        print("{} targets being predicted by ResNet50".format(len(propal_rects)))
        for _, rect in enumerate(propal_rects):
            x, y, h, w = rect
            pred = predict(self.model, self.config.target_size,
                           self.config.top_n, self.image[y:y+h, x:x+w])
            for i in range(self.config.top_n):
                if pred[i][2] > self.config.threshold_for_resnet:
                    preds.append((pred[i], x, y, w, h))
                    if pred[i][1] not in finded_obj_class:
                        finded_obj_class.append(pred[0][1])


        print("{} targets founded by ResNet50".format(len(preds)))
        # 建立一个字典来储存不同类别物体的预测框
        finded_bbox_dict = dict()
        for obj in finded_obj_class:
            finded_bbox_dict[obj] = []
            for (pred, x, y, w, h) in preds:  # pred(nqohxn classname  score),x,y,w,h
                if pred[1] == obj:
                    finded_bbox_dict[obj].append((pred[2], x, y, w, h))  # {classname} [(score,xywh)]

        #self.show_picture(finded_bbox_dict)

        object_bbox_dict = dict()
        for obj in self.config.target_obj:
            object_bbox_dict[obj] = []
            for (pred, x, y, w, h) in preds:    #pred(nqohxn classname  score),x,y,w,h
                if pred[1] == obj:
                    object_bbox_dict[obj].append((pred[2], x, y, w, h)) # {classname} [(score,xywh)]

        #self.show_picture(object_bbox_dict)
        print(object_bbox_dict)
        return object_bbox_dict, finded_bbox_dict

    def show_rect(self, rects):
        plt.figure()
        plt.imshow(self.image)
        ax = plt.gca()
        for x, y, w, h in rects:
            rect = plt.Rectangle((x, y), w, h, color="red", fill=False, linewidth=1)
            ax.add_patch(rect)
        plt.show()
        plt.clf()

    def show_picture(self,bboxes):   #dict
        plt.figure()
        plt.imshow(self.image_save)
        ax = plt.gca()
        for key in bboxes.keys():
            for bbox in bboxes[key]:
                score, x, y, w, h = bbox
                rect = plt.Rectangle((x * self.config.resize_rate, y * self.config.resize_rate),
                                     w * self.config.resize_rate, h * self.config.resize_rate,
                                     color="red", fill=False, linewidth=1)
                ax.add_patch(rect)
                ax.text(x * self.config.resize_rate, y * self.config.resize_rate,
                        "{}".format(key)+'{:.2f}'.format(score), bbox={'facecolor': 'blue', 'alpha': 0.5})
        plt.show()
        plt.savefig("object_detection" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
        plt.clf()

        plt.figure()
        plt.imshow(self.image)
        ax = plt.gca()
        for key in bboxes.keys():
            for bbox in bboxes[key]:
                score, x, y, w, h = bbox
                rect = plt.Rectangle((x, y ),
                                     w , h ,
                                     color="red", fill=False, linewidth=1)
                ax.add_patch(rect)
                ax.text(x , y ,
                        "{}".format(key) + '{:.2f}'.format(score), bbox={'facecolor': 'blue', 'alpha': 0.5})
        plt.show()
        plt.savefig("object_detection" + '.jpg', bbox_inches='tight', dpi=300, format='jpg')
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
    dect = Dector("img/computer.png")
    dect()