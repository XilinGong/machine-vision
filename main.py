import cv2
import easygui
import datetime
import numpy as np
import os
from image_stitching import image_stitching
from objdect import Dector
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    start_time = datetime.datetime.now()

    file_name = 'book'
    img = image_stitching(file_name)
    result = img.stitch()

    end_time = datetime.datetime.now()
    easygui.msgbox(f'图像拼接完毕\n共消耗{end_time-start_time}')

    #result = result.astype(np.uint8)
    cv2.imshow(file_name, result)
    cv2.waitKey(0)

    dect = Dector('src/image/book/result/result.jpg', result)
    dect()
    cv2.destroyAllWindows()
