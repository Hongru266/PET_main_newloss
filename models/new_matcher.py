import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict, Any

class PointsMasksMatcher(nn.Module):
    """
    点与二值掩码匹配器，所有掩码都是numpy数组形式
    """
    
    def __init__(self, weights = None):
        super().__init__()
        # self.forced_pairs = []
        # self.total_cost = 0.0
        # self.background_points = []
        # self.unmatched_masks = []
        self.weights = weights if weights is not None else {
            'direct': 1.0,
            'multiple': 1.0,
            'background': 1.0
        }
    
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

        B = len(targets)
        # print(f'targets[0] keys: {targets[0].keys()}')
        pred_points = outputs["pred_points"]  # 这是一个 tensor 或 numpy 数组
        
        results = []
        pairs_tensor = []
        new_indices = []
        for b in range(B):
            # U = U_batch[b].detach().cpu().numpy()
            U = pred_points[b, :, :]
            # V = V_batch[b].detach().cpu().numpy()
            keys = targets[b]['dicts_keys']
            V = targets[b]['dicts_coords']
            # print(f'Batch {b}: U shape: {U.shape}, V: {V}, keys: {keys}')
            # masks = masks_batch[b].detach().cpu().numpy()
            masks = targets[b]['masks']
            new_masks = self.labelmap_to_masks(masks, keys)  # [N, H, W]
            # print(f'U_batch shape:{U.shape}, target points shape:{V.shape}, target masks shape: {masks.shape}') 
            # N, H, W = new_masks.shape

            forced_pairs, total_cost, unmatched_masks = self._match_single(U, V, new_masks)
            # print(f'Batch {b}: forced_pairs={forced_pairs}, total_cost={total_cost}, unmatched_masks={unmatched_masks}')

            results.append({
                "forced_pairs": forced_pairs,
                "total_cost": total_cost,
                "unmatched_masks": unmatched_masks
            })

            if forced_pairs:
                src = torch.tensor([i for i, _ in forced_pairs], dtype=torch.long)
                tgt = torch.tensor([j for _, j in forced_pairs], dtype=torch.long)
            else:
                # 如果没有匹配，返回空 tensor
                src = torch.tensor([], dtype=torch.long)
                tgt = torch.tensor([], dtype=torch.long)
            new_indices.append((src, tgt))

        # return pairs_tensor
        return new_indices

    def labelmap_to_masks(self, label_map, points_ids, num_classes=None):
        """
        Args:
            label_map: Tensor [H, W]，值为0表示背景，1..N表示不同目标
            points_ids: Tensor [N]，每个点对应的目标ID
            num_classes: int, 可选，如果不指定会自动取最大标签
        
        Returns:
            masks: Tensor [N, H, W]，每个mask是二值 {0,1}
        """
        if num_classes is None:
            num_classes = int(label_map.max().item())  # 自动获取最大标签值
        
        # one-hot 展开: [H, W] -> [H, W, N]
        masks = torch.nn.functional.one_hot(label_map.long(), num_classes=num_classes+1)  # 包含背景
        masks = masks.permute(2, 0, 1).contiguous()  # -> [N+1, H, W]
        # print(f'masks shape:{masks.shape}, num_classes:{num_classes}, points_ids length:{len(points_ids)}')
        points_ids = points_ids + 1  # 因为masks包含背景，点ID需要+1对齐
        valid_mask = (points_ids >= 0) & (points_ids < masks.shape[0])

        # if not valid_mask.all():
        #     print(f"Warning: { (~valid_mask).sum().item() } invalid points_ids detected, will be ignored.")
        points_ids = points_ids[valid_mask]
        # print(f'points_ids shape:{len(points_ids)}, valid_mask shape:{len(valid_mask)}')

        if points_ids.numel() == 0:
            # 返回空 mask 避免报错
            return torch.zeros(0, *label_map.shape, dtype=torch.bool, device=label_map.device)
        
        masks = masks[points_ids]  # 只保留 points_ids 对应的掩码
        # print(f'masks after indexing shape:{masks.shape}')
        # print(f'masks after indexing shape:{masks.shape}')
        
        return masks
    
    # ------------------- 单样本匹配逻辑 -------------------
    def _match_single(self, U, V, masks):
        N, H, W = masks.shape
        num_points = len(U)
        num_masks = masks.shape[0]
        num_gt = len(V)
        # print(f'v:{V}, masks:{masks}')

        # 初始化
        forced_pairs = []
        total_cost = 0.0
        matched_points = set()

        # step1: 判断哪些预测点落在 mask 内
        inside_matrix = self._find_points_in_masks(U, masks, H, W)

        # print(f'inside_matrix shape: {inside_matrix.shape}, inside_matrix: {inside_matrix}, inside_matrix type: {type(inside_matrix)}')

        all_points = set(range(num_points))
        all_in_masks = []
        inside_matrix = inside_matrix.type(torch.int32)
        # print(f'inside_matrix shape: {inside_matrix.shape}, inside_matrix: {inside_matrix}, inside_matrix type: {type(inside_matrix)}')
        mask_vec = (inside_matrix.sum(dim=0) > 0).cpu()   # CPU 上的 bool tensor
        # sum0 = sum0.tolist()
        # print(f'sum:{sum0}, sum0 type:{type(sum0)}')
        all_in_masks = mask_vec.nonzero(as_tuple=False).squeeze(1).tolist()
        all_in_masks = set(all_in_masks)
        # print(f'inside_matrix shape:{inside_matrix.shape}, sum0 shape:{sum0.shape}, sum0:{sum0}, all_in_masks: {all_in_masks}')
        bg_points = all_points - all_in_masks
        avaliable_points = all_in_masks
        # print(f'inside_matrix shape:{inside_matrix.shape}, masks shape:{masks.shape}, points shape:{V.shape}, num_points:{num_points}, num_masks:{num_masks}, num_gt:{num_gt}')

        # --- 遍历GT点 ---
        for j in range(num_gt):
            # 哪些预测点落在当前mask j
            if j >= inside_matrix.shape[0]:
                # print(f'Warning: j={j} exceeds inside_matrix shape {inside_matrix.shape}')
                continue
            inside_points = inside_matrix[j].nonzero(as_tuple=True)[0]  # [K]
            inside_points = [p for p in inside_points.tolist() if p in avaliable_points]

            if len(inside_points) == 0:
                # 没有点 → 随机选一个最小距离
                if not bg_points:
                    continue
                avail_list = list(bg_points)
                dists = torch.cdist(U[avail_list], V[j].unsqueeze(0))[:, 0]  # 计算所有可用预测点与第j个GT点的欧式距离，结果为[len(avail_list), 1]，再取出第一维
                min_idx = torch.argmin(dists).item() #找到距离GT点j最近的预测点在avail_list中的索引
                u = avail_list[min_idx]
                cost = dists[min_idx].item()
                forced_pairs.append((u, j))
                total_cost += cost
                matched_points.add(u)
                bg_points.remove(u)
            elif len(inside_points) == 1:
                # 单点 → 直接匹配
                u = inside_points[0]
                cost = torch.norm(U[u] - V[j]).item()
                forced_pairs.append((u, j))
                total_cost += cost
                matched_points.add(u)
                avaliable_points.remove(u)
            else:
                # 多点 → 选最近的
                # print(f'inside_points: {inside_points}')
                subset = U[inside_points]  # [K, 2]
                dists = torch.cdist(subset, V[j].unsqueeze(0))[:, 0]  # [K]
                min_local = torch.argmin(dists).item()
                u = inside_points[min_local]
                cost = dists[min_local].item()
                forced_pairs.append((u, j))
                total_cost += cost
                matched_points.add(u)
                avaliable_points.remove(u)
                other_points = [p for i, p in enumerate(inside_points) if i != min_local]
                bg_points.update(other_points)  # 其他点放回背景点集合

        matched_masks = {j for _, j in forced_pairs}
        unmatched_masks = list(set(range(num_masks)) - matched_masks)

        # unmatched_masks = list(set(range(num_masks)) - {vj for _, vj in forced_pairs})
        # print(f'matched_points shape:{len(matched_points)}, matched_masks shape:{len(matched_masks)}, unmatched_masks shape:{len(unmatched_masks)}')
        return forced_pairs, total_cost, unmatched_masks
    
    # ------------------- 辅助函数 -------------------
    def _find_points_in_masks(self, U, masks, H, W):
        """
        批量计算哪些点落在每个 mask 内
        U: [num_points, 2] (x,y)，归一化到 [0,1]
        masks: [num_masks, H, W]
        返回: inside_matrix [num_masks, num_points] (bool)
        """
        num_points = U.shape[0]
        num_masks, H, W = masks.shape

        # 1. 映射到像素坐标
        # U_clamped = U.clamp(0.0, 1.0)
        # # print(f'U_clamped min:{U_clamped.min()}, max:{U_clamped.max()}')
        # coords = (U_clamped * torch.tensor([H-1, W-1], device=U.device)).round().long()
        U_clone = U.clone()
        U_clone[:, 0] *= H
        U_clone[:, 1] *= W
        x = U_clone[:, 0]
        y = U_clone[:, 1]

        # 3. 安全性检查
        # if (x < 0).any() or (x >= H ).any():
        #     raise ValueError(f"x 索引越界: min={x.min().item()}, max={x.max().item()}, W={W}")
        # if (y < 0).any() or (y >= W ).any():
        #     raise ValueError(f"y 索引越界: min={y.min().item()}, max={y.max().item()}, H={H}")

        # masks = masks.to(U_clamped.device)
        # print(f'masks shape:{masks.shape}, x min:{x.min()}, x max:{x.max()}, y min:{y.min()}, y max:{y.max()}')

        # 4. 高级索引，计算每个点是否落在每个 mask 内
        x_idx = x.round().long().clamp(0, H - 1)
        y_idx = y.round().long().clamp(0, W - 1)
        inside_matrix = masks[:, x_idx, y_idx] > 0  # [num_masks, num_points]

        return inside_matrix
    
    def _handle_single_point_case(self, U, V, u_idx, v_idx, used_points):
        if u_idx in used_points:
            return 0.0, None
        d = torch.norm(U[u_idx] - V[v_idx])
        cost = d * self.weights['direct']
        used_points.add(u_idx)
        return cost, (u_idx, v_idx)
    
    def _handle_multiple_points_case(self, U, V, pts, v_idx, used_points):
        avail = [u for u in pts if u not in used_points]
        if not avail:
            return 0.0, None
        dists = torch.norm(U[avail] - V[v_idx], dim=1)  # 结果是 [len(avail)] tensor
        min_pos = torch.argmin(dists).item()
        chosen_u = avail[min_pos]
        cost = dists[min_pos] * self.weights['multiple']
        used_points.add(chosen_u)
        return cost, (chosen_u, v_idx)
    
    def _handle_empty_masks(self, U, V, unmatched_masks, background_points, used_points):
        pairs = []
        total_cost = 0.0
        bg_list = list(background_points)
        for j in list(unmatched_masks):
            avail = [u for u in bg_list if u not in used_points]
            if not avail:
                continue
            dists = torch.norm(U[avail] - V[j], dim=1)  # 结果是 [len(avail)] tensor
            min_pos = torch.argmin(dists).item()
            best_u = avail[min_pos]
            cost = dists[min_pos] * self.weights['background']
            pairs.append((best_u, j))
            total_cost += cost
            used_points.add(best_u)
            bg_list.remove(best_u)
        return pairs, total_cost
    

def build_new_matcher(args):
    return PointsMasksMatcher()