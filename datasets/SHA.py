import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import json  # 添加 json 模块导入
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')

# count = 0

class SHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False):
        self.root_path = data_root
        self.train = train  # 先定义 self.train
        self.transform = transform
        self.flip = flip
        self.patch_size = 512
        
        prefix = "train_data" if train else "test_data"
        self.prefix = prefix
        self.img_list = os.listdir(f"{data_root}/{prefix}/images")

        # get image and ground-truth list
        self.gt_list = {}
        self.masks_list = {}
        self.dict_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/images/{img_name}"  
            gt_path = f"{data_root}/{prefix}/ground-truth/GT_{img_name}"
            self.gt_list[img_path] = gt_path.replace("jpg", "mat")
            if self.train:
                masks_path = f"{data_root}/{prefix}/masks/{img_name}"
                dict_path = f"{data_root}/{prefix}/dict/{img_name}"
                self.masks_list[img_path] = masks_path.replace("jpg", "npy")
                self.dict_list[img_path] = dict_path.replace("jpg", "json")
        
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)
    
    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        if self.train:
            masks_path = self.masks_list[img_path]
            dict_path = self.dict_list[img_path]
        else:
            masks_path = None
            dict_path = None
        img, points, masks, dicts = load_data((img_path, gt_path, masks_path, dict_path), self.train)
        points = points.astype(float)

        # image transform
        if self.transform is not None:
            img = self.transform(img)
            # masks = self.transform(masks)
        
        # 确保 img 是 torch 张量
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        elif not isinstance(img, torch.Tensor):
            img = torch.Tensor(img)
        # print(f'after transform: img shape: {img.shape}, points shape: {points.shape}, points[:, 0] max:{points[:, 0].max()}, masks shape: {masks.shape if len(masks) > 0 else "N/A"}, dicts length: {len(dicts)}')
        # random scale
        if self.train:
            scale_range = [0.8, 1.2]           
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            
            # interpolation - 使用新的API
            if scale * min_size > self.patch_size:  
                img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
                points *= scale
                # 只在训练模式下处理 masks 和 dicts
                if len(masks) > 0:  # 检查 masks 不为空
                    # 转换数据类型并转为 PyTorch 张量
                    if masks.dtype == np.uint16:
                        masks = masks.astype(np.uint8)  # 或者使用 np.int32
                    masks = torch.from_numpy(masks)
                    masks = torch.nn.functional.interpolate(masks.unsqueeze(0).unsqueeze(0).float(), scale_factor=scale, mode='nearest').squeeze(0).squeeze(0)
                if dicts:  # 检查 dicts 不为空
                    dicts = {
                        key: (coord[0] * scale, coord[1] * scale) if isinstance(coord, (tuple, list)) and len(coord) >= 2 else coord
                        for key, coord in dicts.items()
                    }
        # print(f'after random scale: img shape: {img.shape}, points shape: {points.shape}, points[:, 0] max:{points[:, 0].max()}, masks shape: {masks.shape if len(masks) > 0 else "N/A"}, dicts length: {len(dicts)}, dicts[:, 0] max: {np.array(list(dicts.values()))[:,0].max() if len(dicts) > 0 else "N/A"}')
        # random crop patch
        patch_h, patch_w = self.patch_size, self.patch_size
        if img.size(1) < patch_h or img.size(2) < patch_w:
            scale_factor = max(patch_h / img.size(1), patch_w / img.size(2))
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode="bilinear", align_corners=False).squeeze(0)
            points = points * scale_factor
            if masks is not None and len(masks) > 0:
                if masks.dtype == np.uint16:
                    masks = masks.astype(np.uint8)  # 或者使用 np.int32
                masks = torch.from_numpy(masks)
                masks = torch.nn.functional.interpolate(masks.unsqueeze(0).unsqueeze(0).float(), scale_factor=scale_factor).squeeze().long()
            dicts = {k: (coord[0]*scale_factor, coord[1]*scale_factor) for k, coord in dicts.items()}
            # print(f'after resize for crop: img shape: {img.shape}, points shape: {points.shape}, points[:, 0] max:{points[:, 0].max()}, masks shape: {masks.shape if len(masks) > 0 else "N/A"}, dicts length: {len(dicts)}')
        if self.train:
            img, points, masks, dicts = random_crop(img, points, masks, dicts, patch_size=self.patch_size)
        # print(f'after random crop: img shape: {img.shape}, points shape: {points.shape}, points[:, 0] max:{points[:, 0].max()}, masks shape: {masks.shape if len(masks) > 0 else "N/A"}, dicts length: {len(dicts)}')
        # random flip 随机翻转进行数据增强
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]
            if len(masks) > 0:  # 只在有 masks 数据时翻转
                if isinstance(masks, np.ndarray):
                    # 处理不支持的数据类型
                    if masks.dtype == np.uint16:
                        masks = masks.astype(np.uint8)
                    masks = torch.from_numpy(masks)
                masks = torch.flip(masks, dims=[1])
            if dicts:  # 只在有 dicts 数据时翻转
                dicts = {
                    key: (coord[0], self.patch_size - coord[1]) if isinstance(coord, (tuple, list)) and len(coord) >= 2 else coord
                    for key, coord in dicts.items()
                }
        
        # target
        target = {}
        dicts_values = list(dicts.values())
        # print(f'img shape:{img.shape}, masks shape:{masks.shape}, dicts[:, 0] max:{np.array(dicts_values)[:, 0].max() if dicts_values else "N/A"}, points shape:{points.shape}, points[:, 0] max:{points[:, 0].max()}, points[:, 1] max:{points[:, 1].max()}')  # 打印前5个点作为示例
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points) #通过计算真实标注点之间的平均最近距离来衡量人群的拥挤程度
            target['density'] = density
            target['masks'] = masks
            
            # 将字典转换为张量格式
            if dicts:  # 字典不为空
                coords = torch.tensor([coord for coord in dicts.values()], dtype=torch.float32)
                # 方案1：如果键是数字字符串，转换为整数张量
                try:
                    keys = torch.tensor([int(key) for key in dicts.keys()], dtype=torch.long)
                except ValueError:
                    # 方案2：如果键不是纯数字，创建索引张量
                    keys = torch.arange(len(dicts), dtype=torch.long)
                target['dicts_coords'] = coords
                target['dicts_keys'] = keys
            else:  # 字典为空
                target['dicts_coords'] = torch.empty(0, 2, dtype=torch.float32)
                target['dicts_keys'] = torch.empty(0, dtype=torch.long)
        else:
            # 测试模式下设置空的 masks 和相关字段
            target['masks'] = torch.empty(0, dtype=torch.uint8)
            target['dicts_coords'] = torch.empty(0, 2, dtype=torch.float32) 
            target['dicts_keys'] = torch.empty(0, dtype=torch.long)
        dicts_coords_values = target['dicts_coords'].numpy() if hasattr(target["dicts_coords"], "numpy") else target["dicts_coords"]
        # print(f'after target: img shape: {img.shape}, points shape: {target["points"].shape}, masks shape: {target["masks"].shape if len(masks) > 0 else "N/A"}, dicts_coords shape: {target["dicts_coords"].shape}, dicts_coords[:, 0] max:{dicts_coords_values[:, 0].max()}, dicts_keys shape: {target["dicts_keys"].shape}')
        if not self.train:
            target['image_path'] = img_path

        # 只在训练模式下且 masks 不为空时计算 offset_map
        if self.train and len(masks) > 0:
            H, W = masks.shape
            unique_labels = np.unique(masks)
            valid_labels = unique_labels[unique_labels != 0]

            if len(valid_labels) == 0:
                # 如果没有有效标签，创建空的 offset_map
                target['offset_map'] = torch.zeros(2, H, W, dtype=torch.float32)
            else:
                # 一次性创建坐标网格
                y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

                # 预分配结果数组
                x_diffs = np.zeros((len(valid_labels), H, W), dtype=np.float32)
                y_diffs = np.zeros((len(valid_labels), H, W), dtype=np.float32)

                # 向量化处理所有标签
                for i, label in enumerate(valid_labels):
                    dict_key = str(int(label - 1))
                    
                    if dict_key in dicts:
                        target_y, target_x = dicts[dict_key]
                        label_mask = (masks == label)
                        
                        # 向量化计算坐标差
                        x_diffs[i][label_mask] = x_coords[label_mask] - target_x
                        y_diffs[i][label_mask] = y_coords[label_mask] - target_y

                x_sum = np.sum(x_diffs, axis=0)  # [H,W] - X方向偏移和
                y_sum = np.sum(y_diffs, axis=0)  # [H,W] - Y方向偏移和

                # 合并为 [2, H, W] 的tensor
                combined_diffs_sum = np.stack([x_sum, y_sum], axis=0)  # [2, H, W]

                # 转换为PyTorch张量
                target['offset_map'] = torch.from_numpy(combined_diffs_sum).float()
        else:
            # 测试模式或没有 masks 时，创建空的 offset_map
            target['offset_map'] = torch.zeros(2, self.patch_size, self.patch_size, dtype=torch.float32)
        
        unique_labels = torch.unique(target['masks'])
        # print(f"SHA: Image: {img_path}, Unique labels in masks: {unique_labels.tolist()}")  # 输出唯一标签
        
        # 检查图像和掩码尺寸是否匹配
        if self.train and len(masks) > 0 and img.shape[-2:] != target['masks'].shape[-2:]:
            print(f"Warning: Image shape {img.shape[-2:]} does not match mask shape {target['masks'].shape[-2:]} for {img_path}")
            print(f"Skipping this sample and trying next one...")
            # 递归调用获取下一个样本（防止无限递归，使用随机索引）
            next_index = random.randint(0, len(self.img_list) - 1)
            if next_index == index:  # 避免重复同一个索引
                next_index = (index + 1) % len(self.img_list)
            return self.__getitem__(next_index)

        #测试gt prob_map和shift_map
        target_masks = target["masks"].float()
        # 将非0值均变为1
        binary_target_masks = (target_masks > 0).float()
        # print(f'binary_target_masks: {binary_target_masks}')
        offset_map = target['offset_map']
        # print(f'offset_map: {offset_map}')
        np.save("0901_binary_target_masks.npy", binary_target_masks.cpu().numpy())
        np.save("0901_offset_map.npy", offset_map.cpu().numpy())
        # print(f'target points shape:{target["points"].shape}, target masks shape:{target["masks"].shape}')

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path, masks_path, dict_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = io.loadmat(gt_path)['image_info'][0][0][0][0][0][:,::-1]  # (N,2) - (x,y)
    
    if train and masks_path is not None and dict_path is not None:
        masks = np.load(masks_path)
        # 对 masks 进行转置
        # masks = masks.T
        
        # 更安全的 JSON 读取方式
        with open(dict_path, 'r', encoding='utf-8') as f:
            dicts = json.load(f)
        
        # 对 dicts 中的坐标进行 (x,y) -> (y,x) 变换
        dicts = {
            key: (coord[1], coord[0]) if isinstance(coord, (tuple, list)) and len(coord) >= 2 else coord
            for key, coord in dicts.items()
        }
    else:
        # 测试模式下返回空的 masks 和 dicts
        masks = np.array([])
        dicts = {}
    dicts_array = np.array(list(dicts.values()))
    # print(f'original: img shape: {img.size}, points shape: {points.shape}, points example:{points[:,0].max()} masks shape: {masks.shape if len(masks) > 0 else "N/A"}, dicts[:, 0] max: {dicts_array[:,0].max() if len(dicts_array) > 0 else "N/A"}')

    return img, points, masks, dicts


def random_crop(img, points, masks, dicts, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    dicts_array = np.array(list(dicts.values()))
    # print(f'img shape before crop: {img.shape}, points shape before crop: {points.shape},points[:, 0] max:{points[:, 0].max()}, masks shape before crop: {masks.shape if len(masks) > 0 else "N/A"}, dicts length before crop: {len(dicts)}, dicts[:, 0] max: {dicts_array[:,0].max() if len(dicts_array) > 0 else "N/A"}')
    
    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)
    # print(f'Crop region - start_h:{start_h}, end_h:{end_h}, start_w:{start_w}, end_w:{end_w}, points before crop: {points.shape[0]}, points after crop: {np.sum(idx)}')
    
    # 优化：使用字典推导式和一次性处理
    # 筛选并直接调整坐标到新的坐标系统
    filtered_dicts = {
        key: (coord[0] - start_h, coord[1] - start_w)
        for key, coord in dicts.items()
        if isinstance(coord, (tuple, list)) and len(coord) >= 2
        and start_h <= coord[0] <= end_h and start_w <= coord[1] <= end_w
    }

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    
    # 处理 masks：先裁剪再转换为 torch 张量（更高效）
    if len(masks) > 0:
        # print(f"Before crop - img shape: {img.shape}, masks shape: {masks.shape}")
        # print(f"Crop region - start_h:{start_h}, end_h:{end_h}, start_w:{start_w}, end_w:{end_w}")
        
        if isinstance(masks, np.ndarray):
            # 先裁剪 numpy 数组
            cropped_masks = masks[start_h:end_h, start_w:end_w]
            # print(f"After crop - cropped_masks shape: {cropped_masks.shape}")
            # 处理不支持的数据类型
            if cropped_masks.dtype == np.uint16:
                cropped_masks = cropped_masks.astype(np.uint8)  # 转换为支持的类型
            # 转换为 torch 张量
            result_masks = torch.from_numpy(cropped_masks)
        else:
            # 如果已经是 torch 张量，直接裁剪
            result_masks = masks[start_h:end_h, start_w:end_w]
            # print(f"After crop - result_masks shape: {result_masks.shape}")
    else:
        result_masks = masks  # 空数组直接返回
    
    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h/imgH, patch_w/imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    
    # 优化：使用字典推导式直接应用缩放因子
    scaled_dicts = {
        key: (coord[0] * fH, coord[1] * fW)
        for key, coord in filtered_dicts.items()
    }
    
    return result_img, result_points, result_masks, scaled_dicts


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path
    if image_set == 'train':
        train_set = SHA(data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = SHA(data_root, train=False, transform=transform)
        return val_set
