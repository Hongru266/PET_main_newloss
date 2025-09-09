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
        self.forced_pairs = []
        self.total_cost = 0.0
        self.background_points = []
        self.unmatched_masks = []
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
        pred_points = outputs["pred_points"]  # 这是一个 tensor 或 numpy 数组
        
        results = []
        pairs_tensor = []
        new_indices = []
        for b in range(B):
            # U = U_batch[b].detach().cpu().numpy()
            U = pred_points[b, :, :]
            # V = V_batch[b].detach().cpu().numpy()
            V = targets[b]['points']
            # masks = masks_batch[b].detach().cpu().numpy()
            masks = targets[b]['masks']
            new_masks = self.labelmap_to_masks(masks)  # [N, H, W]
            # print(f'U_batch shape:{U.shape}, target points shape:{V.shape}, target masks shape: {masks.shape}') 
            N, H, W = new_masks.shape

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

    def labelmap_to_masks(self, label_map, num_classes=None):
        """
        Args:
            label_map: Tensor [H, W]，值为0表示背景，1..N表示不同目标
            num_classes: int, 可选，如果不指定会自动取最大标签
        
        Returns:
            masks: Tensor [N, H, W]，每个mask是二值 {0,1}
        """
        if num_classes is None:
            num_classes = int(label_map.max().item())  # 自动获取最大标签值
        
        # one-hot 展开: [H, W] -> [H, W, N]
        masks = torch.nn.functional.one_hot(label_map.long(), num_classes=num_classes+1)  # 包含背景
        masks = masks.permute(2, 0, 1).contiguous()  # -> [N+1, H, W]
        
        # 去掉背景 (label=0)
        masks = masks[1:]  # [N, H, W]
        
        return masks.float()
    
    # ------------------- 单样本匹配逻辑 -------------------
    def _match_single(self, U, V, masks):
        H, W = masks.shape[1:]
        num_points = len(U)
        num_masks = masks.shape[0]
        num_gt = V.shape[0]

        # 初始化
        forced_pairs = []
        total_cost = 0.0
        used_points = set()

        # step1: 判断哪些预测点落在 mask 内
        inside_matrix = self._find_points_in_masks(U, masks, H, W)
        # print(f'inside_matrix shape:{inside_matrix.shape}, masks shape:{masks.shape}, points shape:{V.shape}, num_points:{num_points}, num_masks:{num_masks}, num_gt:{num_gt}')

        # --- 遍历GT点 ---
        for j in range(num_gt):
            # 哪些预测点落在当前mask j
            if j >= inside_matrix.shape[0]:
                # print(f'Warning: j={j} exceeds inside_matrix shape {inside_matrix.shape}')
                continue
            inside_points = inside_matrix[j].nonzero(as_tuple=True)[0]  # [K]

            if len(inside_points) == 0:
                # 没有点 → 随机选一个最小距离
                dists = torch.cdist(U, V[j].unsqueeze(0))[:, 0]  # [num_points]
                min_idx = torch.argmin(dists).item()
                cost = dists[min_idx].item()
                forced_pairs.append((min_idx, j))
                total_cost += cost
            elif len(inside_points) == 1:
                # 单点 → 直接匹配
                u = inside_points.item()
                cost = torch.norm(U[u] - V[j]).item()
                forced_pairs.append((u, j))
                total_cost += cost
            else:
                # 多点 → 选最近的
                subset = U[inside_points]  # [K, 2]
                dists = torch.cdist(subset, V[j].unsqueeze(0))[:, 0]  # [K]
                min_local = torch.argmin(dists).item()
                u = inside_points[min_local].item()
                cost = dists[min_local].item()
                forced_pairs.append((u, j))
                total_cost += cost

        matched_masks = {j for _, j in forced_pairs}
        unmatched_masks = list(set(range(num_masks)) - matched_masks)

        # unmatched_masks = list(set(range(num_masks)) - {vj for _, vj in forced_pairs})
        return forced_pairs, total_cost, unmatched_masks
    
    # ------------------- 辅助函数 -------------------
    def _find_points_in_masks(self, U, masks, H, W):
        """
        改进版本: 批量计算哪些点落在每个mask里
        U: [num_points, 2] (x,y)
        masks: [num_masks, H, W]
        返回: inside_matrix [num_masks, num_points] (bool)
        """
        num_points = U.shape[0]
        num_masks, H, W = masks.shape

        coords = U.round().long()  # 四舍五入取整
        x = coords[:, 0].clamp(0, W - 1)  # [num_points]
        y = coords[:, 1].clamp(0, H - 1)

        # 取每个mask在这些点上的值，向量化完成
        inside_matrix = masks[:, y, x] > 0  # [num_masks, num_points]
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