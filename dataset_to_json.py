import os
import sys

json_root = '/home/huluwa/data/img_lay/pro_json'
json_names = os.listdir(json_root)

for name in json_names:
    if '.jpg' in name:
        continue
    json_path = os.path.join(json_root, name)
    cmd_str = 'labelme_json_to_dataset' + ' ' + json_path
    print(cmd_str)
    os.system(cmd_str)

