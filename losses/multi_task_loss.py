import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_seg=1.0,
        lambda_inter=1.0,
        lambda_orient=1.0,
        lambda_anchor=0.0,
        lambda_topo=0.0,
        seg_pos_weight=1.0,
        inter_pos_weight=1.0,
        use_dynamic_pos_weight=True,
        use_dynamic_seg_pos_weight=True,
        use_dynamic_inter_pos_weight=False,
        min_pos_weight=5.0,
        max_pos_weight=400.0,
        orient_num_bins=0,
        orient_focal_gamma=2.0,
        lambda_orient_smooth=0.1,
    ):
        super(MultiTaskLoss, self).__init__()
        self.lambda_seg = lambda_seg
        self.lambda_inter = lambda_inter
        self.lambda_orient = lambda_orient
        self.lambda_anchor = lambda_anchor
        self.lambda_topo = lambda_topo
        self.orient_num_bins = int(orient_num_bins)
        self.orient_focal_gamma = float(orient_focal_gamma)
        self.lambda_orient_smooth = float(lambda_orient_smooth)
        self.seg_pos_weight = float(seg_pos_weight)
        self.inter_pos_weight = float(inter_pos_weight)
        self.use_dynamic_pos_weight = bool(use_dynamic_pos_weight)
        self.use_dynamic_seg_pos_weight = bool(use_dynamic_seg_pos_weight)
        self.use_dynamic_inter_pos_weight = bool(use_dynamic_inter_pos_weight)
        self.min_pos_weight = float(min_pos_weight)
        self.max_pos_weight = float(max_pos_weight)

        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def dice_loss_with_logits(logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        intersection = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3))
        union = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3)) + eps
        dice = 1.0 - (intersection + eps) / union
        return dice.mean()

    @staticmethod
    def masked_cosine_loss(pred_xy, gt_xy, mask, eps=1e-6):
        pred_norm = torch.sqrt(torch.sum(pred_xy * pred_xy, dim=1, keepdim=True) + eps)
        gt_norm = torch.sqrt(torch.sum(gt_xy * gt_xy, dim=1, keepdim=True) + eps)
        pred_unit = pred_xy / pred_norm
        gt_unit = gt_xy / gt_norm
        cos = torch.sum(pred_unit * gt_unit, dim=1, keepdim=True)
        cos_loss = (1.0 - cos) * mask
        denom = torch.sum(mask) + eps
        return torch.sum(cos_loss) / denom

    def forward(self, outputs, targets):
        seg_logits = outputs["segmentation"]
        inter_logits = outputs["intersection"]
        orient_logits = outputs["orientation"]

        seg_gt = targets["mask"]
        inter_gt = targets["intersection"]
        orient_gt = targets["orientation"]

        if self.use_dynamic_pos_weight and self.use_dynamic_seg_pos_weight:
            pos = torch.sum(seg_gt)
            neg = seg_gt.numel() - pos
            dyn_seg = (neg / (pos + 1e-6)).clamp(self.min_pos_weight, self.max_pos_weight)
            pw_seg = dyn_seg.to(device=seg_logits.device, dtype=seg_logits.dtype)
        else:
            pw_seg = torch.tensor(self.seg_pos_weight, device=seg_logits.device, dtype=seg_logits.dtype)
        seg_bce = F.binary_cross_entropy_with_logits(
            seg_logits, seg_gt, pos_weight=pw_seg, reduction="mean"
        )
        seg_dice = self.dice_loss_with_logits(seg_logits, seg_gt)
        seg_loss = seg_bce + seg_dice

        if self.use_dynamic_pos_weight and self.use_dynamic_inter_pos_weight:
            inter_pos = torch.sum(inter_gt > 0.5)
            inter_neg = inter_gt.numel() - inter_pos
            dyn_inter = (inter_neg / (inter_pos + 1e-6)).clamp(self.min_pos_weight, self.max_pos_weight)
            pw_inter = dyn_inter.to(device=inter_logits.device, dtype=inter_logits.dtype)
        else:
            pw_inter = torch.tensor(self.inter_pos_weight, device=inter_logits.device, dtype=inter_logits.dtype)
        inter_loss = F.binary_cross_entropy_with_logits(
            inter_logits, inter_gt, pos_weight=pw_inter, reduction="mean"
        )

        gt_dxdy = orient_gt[:, 0:2, :, :]
        gt_conf = orient_gt[:, 2:3, :, :]

        if self.orient_num_bins > 0 and orient_logits.shape[1] == self.orient_num_bins:
            from losses.enhanced_orient_loss import enhanced_orient_losses
            from utils.orient_bins import logits_to_expected_dxdy

            focal, smooth, orient_loss = enhanced_orient_losses(
                orient_logits,
                orient_gt,
                self.orient_num_bins,
                focal_gamma=self.orient_focal_gamma,
                lambda_smooth=self.lambda_orient_smooth,
            )
            exp_xy = logits_to_expected_dxdy(orient_logits)
            pred_norm = torch.sqrt(torch.sum(exp_xy * exp_xy, dim=1, keepdim=True) + 1e-6)
            pred_dxdy = exp_xy / pred_norm
            orient_vec_loss = focal
            orient_conf_loss = smooth
            anchor_loss = self.anchor_consistency_loss(
                torch.sigmoid(inter_logits), pred_dxdy, (gt_conf > 0.5).float()
            )
        else:
            pred_dxdy = torch.tanh(orient_logits[:, 0:2, :, :])
            pred_conf = orient_logits[:, 2:3, :, :]
            gt_norm = torch.sqrt(torch.sum(gt_dxdy * gt_dxdy, dim=1, keepdim=True) + 1e-6)
            valid_mask = ((gt_conf > 0.5) & (gt_norm > 0.5)).float()
            orient_vec_loss = self.masked_cosine_loss(pred_dxdy, gt_dxdy, valid_mask)
            orient_conf_loss = self.bce(pred_conf, gt_conf)
            orient_loss = orient_vec_loss + orient_conf_loss
            anchor_loss = self.anchor_consistency_loss(
                torch.sigmoid(inter_logits), pred_dxdy, (gt_conf > 0.5).float()
            )

        topo_loss = torch.zeros(1, device=seg_logits.device, dtype=seg_logits.dtype).squeeze()
        total_loss = (
            self.lambda_seg * seg_loss
            + self.lambda_inter * inter_loss
            + self.lambda_orient * orient_loss
            + self.lambda_anchor * anchor_loss
            + self.lambda_topo * topo_loss
        )

        return {
            "total_loss": total_loss,
            "segmentation_loss": seg_loss,
            "seg_bce": seg_bce.detach(),
            "seg_dice": seg_dice.detach(),
            "intersection_loss": inter_loss,
            "orientation_loss": orient_loss,
            "orientation_vec_loss": orient_vec_loss,
            "orientation_conf_loss": orient_conf_loss,
            "anchor_loss": anchor_loss,
            "topology_loss": topo_loss,
            "seg_pos_weight_used": pw_seg.detach(),
            "inter_pos_weight_used": pw_inter.detach(),
        }

    @staticmethod
    def anchor_consistency_loss(inter_prob, pred_dxdy, road_mask):
        # 交叉口锚定一致性：热图拉普拉斯应与方向场散度互补（趋向于汇聚）
        lap_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=inter_prob.dtype,
            device=inter_prob.device,
        ).view(1, 1, 3, 3)
        dx_kernel = torch.tensor(
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            dtype=inter_prob.dtype,
            device=inter_prob.device,
        ).view(1, 1, 3, 3) / 6.0
        dy_kernel = torch.tensor(
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            dtype=inter_prob.dtype,
            device=inter_prob.device,
        ).view(1, 1, 3, 3) / 6.0

        inter_lap = F.conv2d(inter_prob, lap_kernel, padding=1)
        div_x = F.conv2d(pred_dxdy[:, 0:1], dx_kernel, padding=1)
        div_y = F.conv2d(pred_dxdy[:, 1:2], dy_kernel, padding=1)
        divergence = div_x + div_y
        return torch.mean(road_mask * torch.square(inter_lap + divergence))
