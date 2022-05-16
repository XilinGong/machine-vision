import cv2
import easygui
import datetime
import numpy as np
import os
from image_stitching import image_stitching
from objdect import Dector
from config import main_config as config


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    file_name = config.filename
    start_time = datetime.datetime.now()

    img = image_stitching(file_name)
    result = img.stitch()

    end_time = datetime.datetime.now()
    easygui.msgbox(f'图像拼接完毕\n共消耗{end_time-start_time}')

    #result = result.astype(np.uint8)
    cv2.imshow(file_name, result)
    cv2.waitKey(0)

    dect = Dector(config.filename_obj, result)
    dect()
    cv2.destroyAllWindows()
