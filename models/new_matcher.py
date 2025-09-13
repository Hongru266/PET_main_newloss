import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment

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
        points_ids = points_ids + 1  # 因为masks包含背景，点ID需要+1对齐
        valid_mask = (points_ids >= 0) & (points_ids < masks.shape[0])

        points_ids = points_ids[valid_mask]
       
        if points_ids.numel() == 0:
            # 返回空 mask 避免报错
            return torch.zeros(0, *label_map.shape, dtype=torch.bool, device=label_map.device)
        
        masks = masks[points_ids]  # 只保留 points_ids 对应的掩码
        
        return masks
    
    # ------------------- TopK+贪心匹配逻辑 -------------------
    def _match_single_topk_greedy(self, U: torch.Tensor, V: torch.Tensor, masks: torch.Tensor, K: int = 5):
        """
        基于 Top-K 候选 + 全局贪心分配的匹配策略（方案B）。
        返回与 _match_single 一致：
            forced_pairs: list of (u_idx, v_idx)
            total_cost: float (匹配对的欧氏距离和)
            unmatched_masks: list[int], 未匹配的 mask 索引
        Args:
            U: 预测点坐标 [P, 2] (归一化到 [0,1])
            V: GT 点坐标 [G, 2]
            masks: 掩码张量 [M, H, W] (bool)，前 G 行与 V 对应
            K: 每个 GT 保留的候选预测点数
        """
        device = U.device
        P = U.shape[0]
        G = V.shape[0]
        M = masks.shape[0]

        if P == 0 or G == 0:
            return [], 0.0, list(range(M))

        # --- 1. inside_matrix: [M, P] (mask m vs point p) ---
        H, W = masks.shape[1], masks.shape[2]
        inside_matrix = self._find_points_in_masks(U, masks, H, W)  # [M, P]
        if inside_matrix.shape[0] < G:
            pad = torch.zeros((G - inside_matrix.shape[0], P), dtype=torch.bool, device=device)
            inside_matrix = torch.cat([inside_matrix, pad], dim=0)
        inside_gp = inside_matrix[:G, :]  # [G, P]

        cand_dists = []
        cand_pidx = []
        cand_gidx = []

        # --- 2. 为每个 GT 选择 top-K 候选点 ---
        for g in range(G):
            pts_idx = torch.nonzero(inside_gp[g], as_tuple=True)[0]  # mask g 内的候选点索引
            if pts_idx.numel() == 0:
                continue  # mask 内没有预测点 -> unmatched
            diff = U[pts_idx] - V[g].unsqueeze(0)  # [n_g, 2]
            d = torch.norm(diff, dim=1)  # [n_g]
            k = min(K, d.shape[0])
            vals, idxs = torch.topk(d, k=k, largest=False)  # 取前k个最小距离
            sel_pts = pts_idx[idxs]  # 预测点索引
            cand_dists.append(vals)
            cand_pidx.append(sel_pts)
            cand_gidx.append(torch.full((k,), g, dtype=torch.long, device=device))

        if len(cand_dists) == 0:
            # 所有 GT 都没有候选 -> 全部 unmatched
            return [], 0.0, list(range(M))

        cand_dists = torch.cat(cand_dists)  # [N_candidates]
        cand_pidx = torch.cat(cand_pidx)
        cand_gidx = torch.cat(cand_gidx)

        # --- 3. 按距离排序并贪心分配 ---
        order = torch.argsort(cand_dists)  # 从小到大排序
        used_points = torch.zeros((P,), dtype=torch.bool, device=device)
        used_gts = torch.zeros((G,), dtype=torch.bool, device=device)
        forced_pairs = []
        total_cost = 0.0

        for idx in order.tolist():
            p = int(cand_pidx[idx].item())
            g = int(cand_gidx[idx].item())
            if not used_points[p] and not used_gts[g]:
                forced_pairs.append((p, g))
                total_cost += float(cand_dists[idx].item())
                used_points[p] = True
                used_gts[g] = True

        # --- 4. 统计未匹配的 mask ---
        matched_masks = {g for _, g in forced_pairs}
        unmatched_masks = list(set(range(M)) - matched_masks)

        return forced_pairs, total_cost, unmatched_masks

    
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

        # step1: 判断哪些预测点落在 mask 内
        inside_matrix = self._find_points_in_masks(U, masks, H, W)

        all_points = set(range(num_points))
        bg_points = set(range(num_points))
        matched_masks = set()
        matched_points = set()

        unmatched_gt = []

        # --- 遍历GT点 ---
        for j in range(num_gt):
            # 哪些预测点落在当前mask j
            if j >= inside_matrix.shape[0]:
                unmatched_gt.append(j)
                # print(f'Warning: j={j} exceeds inside_matrix shape {inside_matrix.shape}')
                continue
            inside_points = inside_matrix[j].nonzero(as_tuple=True)[0]  # [K]
            inside_points = [p for p in inside_points.tolist() if p in bg_points]

            if len(inside_points) == 0:
                # 没有点 → 先不处理
                unmatched_gt.append(j)
            elif len(inside_points) == 1:
                # 单点 → 直接匹配
                u = inside_points[0]
                cost = torch.norm(U[u] - V[j]).item()
                forced_pairs.append((u, j))
                total_cost += cost
                matched_points.add(u)
                matched_masks.add(j)
                bg_points.remove(u)
            else:
                # 多点 → 选最近的
                subset = U[inside_points]  # [K, 2]
                dists = torch.cdist(subset, V[j].unsqueeze(0))[:, 0]  # [K]
                min_local = torch.argmin(dists).item()
                u = inside_points[min_local]
                cost = dists[min_local].item()
                forced_pairs.append((u, j))
                total_cost += cost
                matched_points.add(u)
                matched_masks.add(j)
                bg_points.remove(u)

        # Step3: 剩余的GT与背景点，用hoo
        if unmatched_gt and bg_points:
            bg_list = list(bg_points)
            U_bg = U[bg_list]
            V_gt = V[unmatched_gt]

            cost_matrix = torch.cdist(U_bg, V_gt)  # [Nb, Ng]

            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            for r, c in zip(row_ind, col_ind):
                u = bg_list[r]
                j = unmatched_gt[c]
                cost = cost_np[r, c]
                if cost < 1e5:
                    forced_pairs.append((u, j))
                    total_cost += cost
                    matched_points.add(u)
                    matched_masks.add(j)
                    bg_points.remove(u)

        unmatched_masks = list(set(range(num_masks)) - matched_masks)

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