import cv2
import easygui
import datetime
import numpy as np
from image_stitching import image_stitching
if __name__ == '__main__':
    start_time = datetime.datetime.now()

    file_name = 'testing_hard'
    img = image_stitching(file_name)
    result = img.stitch()

    end_time = datetime.datetime.now()
    easygui.msgbox(f'图像拼接完毕\n共消耗{end_time-start_time}')

    result = result.astype(np.uint8)
    cv2.imshow(file_name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()