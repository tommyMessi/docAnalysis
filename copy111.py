import os
import shutil

from_root = '/home/huluwa/data/img_lay/train_data_min2/labelme_json/'
to_root = '/home/huluwa/data/img_lay/train_data_min2/cv2_mask/'
for dir_name in os.listdir(from_root):
    pic_name = dir_name[:-5] + '.png'
    from_dir = from_root + dir_name + '/label.png'
    to_dir = to_root + pic_name
    shutil.copy(from_dir, to_dir)

    print(from_dir)
    print(to_dir)