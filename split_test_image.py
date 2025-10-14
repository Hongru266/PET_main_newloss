import os
import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image
import json


def split_test_images_with_mat(
    img_dir, mat_dir, save_img_dir, save_mat_dir,
    crop_size=512, pad_value=0
):
    """
    同时切分 test 图像和对应的 .mat 文件（真实点标注）
    """
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mat_dir, exist_ok=True)
    
    patch_info_list = []
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    image_list = sorted(img_files)

    count=0

    for img_id, img_name in enumerate(image_list):
        # ---- 1. 读取图像 ----
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        # ---- 2. 读取对应的 .mat 文件 ----
        mat_path = os.path.join(mat_dir, os.path.splitext(img_name)[0] + '_ann.mat')
        mat_data = loadmat(mat_path)
        points = mat_data['annPoints']  # [N, 2], (x, y)
        
        # ---- 3. padding 图像 ----
        new_w = ((w - 1) // crop_size + 1) * crop_size
        new_h = ((h - 1) // crop_size + 1) * crop_size
        padded_img = Image.new('RGB', (new_w, new_h), color=(pad_value, pad_value, pad_value))
        padded_img.paste(image, (0, 0))

        # ---- 4. 切块 & 生成对应的 .mat ----
        # patch_id = 0
        for top in range(0, new_h, crop_size):
            for left in range(0, new_w, crop_size):
                patch = padded_img.crop((left, top, left + crop_size, top + crop_size))
                count += 1
                # patch_id += 1

                # 当前patch的区域范围
                x_min, x_max = left, left + crop_size
                y_min, y_max = top, top + crop_size

                # ---- 4.1 找到在这个区域内的GT点 ----
                in_patch_mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                                (points[:, 1] >= y_min) & (points[:, 1] < y_max)
                points_in_patch = points[in_patch_mask].copy()
                
                # 坐标平移到patch内部坐标系
                points_in_patch[:, 0] -= x_min
                points_in_patch[:, 1] -= y_min

                # ---- 4.2 保存patch图像 ----
                img_save_name = f"img_{count:04d}.png"
                img_save_path = os.path.join(save_img_dir, img_save_name)
                patch.save(img_save_path)

                # ---- 4.3 保存对应的mat文件 ----
                mat_save_path = os.path.join(save_mat_dir, f"img_{count:04d}_ann.mat")
                savemat(mat_save_path, {'annPoints': points_in_patch})

                # ---- 4.4 保存索引信息 ----
                patch_info_list.append({
                    'orig_img_id': img_id,
                    'orig_img_name': img_name,
                    'patch_id': count,
                    'x_min': x_min, 'y_min': y_min,
                    'orig_h': h, 'orig_w': w,
                    'patch_h': crop_size, 'patch_w': crop_size,
                    'num_gt_points': len(points_in_patch)
                })
                

    return patch_info_list

image_path = './data/UCF-QNRF/Test_original/*.jpg'
mat_path = './data/UCF-QNRF/Test_original/*.mat'
save_image_path = './data/UCF-QNRF/Test_split'
save_mat_path = './data/UCF-QNRF/Test_split'

info = split_test_images_with_mat(
    img_dir='./data/UCF-QNRF/Test_original',
    mat_dir='./data/UCF-QNRF/Test_original',
    save_img_dir=save_image_path,
    save_mat_dir=save_mat_path,
    crop_size=512,
    pad_value=0
)

# 保存到指定路径
with open('./data/UCF-QNRF/Test_split/info.json', 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)