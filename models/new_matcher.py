import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict, Any

class PointsMasksMatcher(nn.Module):
    """
    点与二值掩码匹配器，所有掩码都是numpy数组形式
    """
    
    def __init__(self):
        super().__init__()
        self.forced_pairs = []
        self.total_cost = 0.0
        self.background_points = []
        self.unmatched_masks = []
    
    def forward(self, outputs: Dict[str, Any], targets: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], float]:
        """
        适配outputs和targets的forward方法
        
        Args:
            outputs: 模型输出，包含pred_points等
            targets: 目标数据列表，每个元素包含points, masks等
            
        Returns:
            pairs: 匹配对列表 [(u_idx, v_idx), ...]
            total_cost: 总匹配代价
        """
        # 重置状态
        self.forced_pairs = []
        self.total_cost = 0.0
        self.background_points = []
        self.unmatched_masks = []
        
        # 提取预测点和目标数据（假设处理batch中的第一个样本）
        U = self._extract_predicted_points(outputs)  # 预测点
        target_data = targets[0]  # 第一个目标样本
        V = target_data['points']  # 真实点
        masks = target_data['masks']  # 掩码
        
        # 输入验证
        self._validate_inputs(U, V, masks, outputs, target_data)
        
        # 获取图像尺寸
        height, width = self._get_image_shape(outputs, target_data)
        
        # 存储每个mask包含的预测点索引
        mask_points = [[] for _ in range(len(V))]
        points_in_any_mask = np.zeros(len(U), dtype=bool)
        
        # 检查每个预测点落在哪些mask中
        self._find_points_in_masks(U, masks, mask_points, points_in_any_mask, height, width)
        
        # 识别背景点（不在任何mask中的点）
        self.background_points = [i for i in range(len(U)) if not points_in_any_mask[i]]
        
        # 执行匹配策略
        self._apply_matching_strategy(U, V, mask_points)
        
        # 识别未匹配的mask
        self._identify_unmatched_masks(mask_points)
        
        # 按目标点索引排序输出
        self.forced_pairs.sort(key=lambda x: x[1])
        return self.forced_pairs, self.total_cost
    
    def _extract_predicted_points(self, outputs: Dict[str, Any]) -> np.ndarray:
        """从outputs中提取预测点"""
        if 'pred_points' in outputs:
            # 假设pred_points形状为 [batch_size, num_points, 2]
            pred_points = outputs['pred_points']
            if isinstance(pred_points, (list, tuple)):
                pred_points = pred_points[0]  # 取第一个batch
            if hasattr(pred_points, 'detach'):
                pred_points = pred_points.detach().cpu().numpy()
            return pred_points
        else:
            raise ValueError("outputs中缺少pred_points")
        
    def _get_image_shape(self, outputs: Dict[str, Any], target_data: Dict[str, Any]) -> Tuple[int, int]:
        """获取图像尺寸"""
        # 优先从outputs中获取
        if 'img_shape' in outputs:
            img_shape = outputs['img_shape']
            if isinstance(img_shape, (list, tuple)):
                return tuple(img_shape[:2])  # (height, width)
        
        # 其次从targets中获取（如果有）
        if 'masks' in target_data and len(target_data['masks']) > 0:
            mask_shape = target_data['masks'][0].shape
            return mask_shape  # (height, width)
        
        # 最后从offset_map获取（如果有）
        if 'offset_map' in target_data:
            offset_shape = target_data['offset_map'].shape[:2]
            return offset_shape
        
        raise ValueError("无法确定图像尺寸")
    
    def _validate_inputs(self, U: np.ndarray, V: np.ndarray, masks: List[np.ndarray], 
                        outputs: Dict[str, Any], target_data: Dict[str, Any]) -> None:
        """验证输入数据"""
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError(f"预测点U必须是(n, 2)的数组，当前形状: {U.shape}")
        if V.ndim != 2 or V.shape[1] != 2:
            raise ValueError(f"真实点V必须是(n, 2)的数组，当前形状: {V.shape}")
        if not masks:
            raise ValueError("掩码列表不能为空")
        if not all(isinstance(mask, np.ndarray) and mask.ndim == 2 for mask in masks):
            raise ValueError("所有掩码必须是二维numpy数组")
        
        # 检查所有掩码尺寸是否一致
        first_shape = masks[0].shape
        if any(mask.shape != first_shape for mask in masks):
            raise ValueError("所有掩码必须具有相同的尺寸")
        
        # 检查点数一致性
        if len(V) != len(masks):
            raise ValueError(f"真实点数量({len(V)})与掩码数量({len(masks)})不一致")
    
    def _find_points_in_masks(self, U: np.ndarray, masks: List[np.ndarray], 
                             mask_points: List[List[int]], points_in_any_mask: np.ndarray,
                             height: int, width: int) -> None:
        """找出每个mask中包含的预测点"""
        # 将浮点坐标转换为像素坐标
        points_pixel = np.round(U).astype(int)
        valid_x = np.clip(points_pixel[:, 0], 0, width - 1)
        valid_y = np.clip(points_pixel[:, 1], 0, height - 1)
        
        for j, mask in enumerate(masks):
            # 直接检查掩码值
            in_mask = mask[valid_y, valid_x] > 0
            indices_in_mask = np.where(in_mask)[0]
            mask_points[j].extend(indices_in_mask)
            points_in_any_mask[indices_in_mask] = True
    
    def _apply_matching_strategy(self, U: np.ndarray, V: np.ndarray, 
                                mask_points: List[List[int]]) -> None:
        """应用匹配策略"""
        # 处理有预测点的mask
        for j, points_in_mask in enumerate(mask_points):
            num_points = len(points_in_mask)
            
            if num_points == 1:
                # 情况1：有且仅有一个预测点，直接配对
                self._handle_single_point_case(U, V, points_in_mask[0], j)
                
            elif num_points > 1:
                # 情况2：有多个预测点，选择距离最近的点
                self._handle_multiple_points_case(U, V, points_in_mask, j)
        
        # 处理无预测点的mask（情况3）
        self._handle_empty_masks(U, V, mask_points)
    
    def _handle_single_point_case(self, U: np.ndarray, V: np.ndarray, 
                                 u_idx: int, v_idx: int) -> None:
        """处理单个预测点的情况"""
        distance = np.linalg.norm(U[u_idx] - V[v_idx])
        weighted_cost = distance * self.weights['direct']
        self.forced_pairs.append((u_idx, v_idx))
        self.total_cost += weighted_cost
    
    def _handle_multiple_points_case(self, U: np.ndarray, V: np.ndarray, 
                                    points_in_mask: List[int], v_idx: int) -> None:
        """处理多个预测点的情况"""
        distances = [np.linalg.norm(U[u_idx] - V[v_idx]) for u_idx in points_in_mask]
        min_idx = np.argmin(distances)
        weighted_cost = distances[min_idx] * self.weights['multiple']
        self.forced_pairs.append((points_in_mask[min_idx], v_idx))
        self.total_cost += weighted_cost
    
    def _handle_empty_masks(self, U: np.ndarray, V: np.ndarray, 
                           mask_points: List[List[int]]) -> None:
        """处理无预测点的mask"""
        # 找出所有不在任何mask中的预测点（背景点）
        all_matched_points = set()
        for pairs in mask_points:
            all_matched_points.update(pairs)
        
        background_points = set(range(len(U))) - all_matched_points
        
        if not background_points:
            return
            
        background_points_list = list(background_points)
        background_U = U[background_points_list]
        
        # 为每个无预测点的mask找到最近的背景点
        for j, points_in_mask in enumerate(mask_points):
            if len(points_in_mask) == 0:  # 无预测点的mask
                if background_points_list:
                    # 计算到所有背景点的距离
                    distances = np.linalg.norm(background_U - V[j], axis=1)
                    min_idx = np.argmin(distances)
                    best_u_idx = background_points_list[min_idx]
                    
                    weighted_cost = distances[min_idx] * self.weights['background']
                    self.forced_pairs.append((best_u_idx, j))
                    self.total_cost += weighted_cost
                    
                    # 从背景点中移除已使用的点
                    background_points_list.pop(min_idx)
                    background_U = np.delete(background_U, min_idx, axis=0)
    
    def _identify_unmatched_masks(self, mask_points: List[List[int]]) -> None:
        """识别未匹配的mask"""
        matched_masks = set(j for _, j in self.forced_pairs)
        self.unmatched_masks = [j for j in range(len(mask_points)) if j not in matched_masks]
    
    def get_match_info(self) -> Dict[str, Any]:
        """获取匹配的详细信息"""
        return {
            'pairs': self.forced_pairs,
            'total_cost': self.total_cost,
            'background_points': self.background_points,
            'unmatched_masks': self.unmatched_masks,
            'weights': self.weights
        }

def build_new_matcher(args):
    return PointsMasksMatcher()