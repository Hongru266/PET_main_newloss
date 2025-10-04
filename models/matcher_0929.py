import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher with additional mask-based constraint (改进版).
    """
    def __init__(self, cost_class: float = 1, cost_point: float = 1, cost_mask: float = 1, sigma: float = 10.0):
        """
        Params:
            cost_class: 权重 - 分类代价
            cost_point: 权重 - 点坐标 L2 距离
            cost_mask:  权重 - 掩码区域约束
            sigma:      高斯惩罚控制参数 (越大 → 惩罚范围越宽)
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.cost_mask = cost_mask
        self.sigma = sigma  # <<< 新增
        assert cost_class != 0 or cost_point != 0 or cost_mask != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, **kwargs):
        """
        Performs the matching with mask constraints.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs*num_queries, 2]
        out_points = outputs["pred_points"].flatten(0, 1)            # [bs*num_queries, 2]

        # concat target info
        tgt_ids = torch.cat([v["labels"] for v in targets])          # all GT labels
        tgt_points = torch.cat([v["dicts_coords"] for v in targets]) # [sum_gt, 2]

        # 生成所有 GT 的掩码列表
        masks_list = []
        for i in range(bs):
            tgt_masks = targets[i]['masks']       # [num_gt, H, W]
            tgt_keys = targets[i]['dicts_keys']  # [num_gt]
            masks = self.labelmap_to_masks(tgt_masks, tgt_keys)
            masks_list.append(masks)
        tgt_masks = torch.cat(masks_list, dim=0)  # [sum_gt, H, W]

        # classification cost
        cost_class = -out_prob[:, tgt_ids]

        # point distance cost
        img_h, img_w = outputs["img_shape"]
        out_points_abs = out_points.clone()
        out_points_abs[:, 0] *= img_h
        out_points_abs[:, 1] *= img_w
        cost_point = torch.cdist(out_points_abs, tgt_points, p=2)  # [bs*num_queries, sum_gt]

        # mask-based cost (改进版)
        cost_mask = torch.zeros_like(cost_point)

        # 遍历每个 GT mask
        for j, tgt_mask in enumerate(tgt_masks):  
            # 预测点坐标
            xs = out_points_abs[:, 0].long().clamp(0, img_h - 1)
            ys = out_points_abs[:, 1].long().clamp(0, img_w - 1)

            # --- 修改点 1: 原来是二值 penalty，现在区分 mask 内外 ---
            inside_mask = tgt_mask[ys, xs].float()  # <<< 修复 index bug: 应该先 y 后 x

            # --- 修改点 2: 高斯惩罚 (鼓励靠近 GT 点中心) ---
            dist2 = ((out_points_abs - tgt_points[j]) ** 2).sum(dim=1)  # 欧式距离平方
            gaussian_penalty = torch.exp(-dist2 / (2 * self.sigma ** 2))  # [0,1]

            # --- 修改点 3: 组合规则 ---
            # mask 外部 → 大惩罚 (10)
            # mask 内部 → 根据距离惩罚 (1 - gaussian_penalty)
            penalty = (1 - inside_mask) * 10.0 + inside_mask * (1 - gaussian_penalty)

            cost_mask[:, j] = penalty

        # 最终 cost
        C = self.cost_point * cost_point + self.cost_class * cost_class + self.cost_mask * cost_mask
        C = C.view(bs, num_queries, -1).cpu()

        # split by batch
        sizes = [len(v["points"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

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
        
        masks = torch.nn.functional.one_hot(label_map.long(), num_classes=num_classes+1)  # 包含背景
        masks = masks.permute(2, 0, 1).contiguous()  # -> [N+1, H, W]
        points_ids = points_ids + 1  # 因为masks包含背景，点ID需要+1对齐
        valid_mask = (points_ids >= 0) & (points_ids < masks.shape[0])

        points_ids = points_ids[valid_mask]
       
        if points_ids.numel() == 0:
            return torch.zeros(0, *label_map.shape, dtype=torch.bool, device=label_map.device)
        
        masks = masks[points_ids]  # 只保留 points_ids 对应的掩码
        
        return masks

def build_0929_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point, cost_mask=args.set_cost_mask)
