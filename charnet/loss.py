import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, pred_char_fg, pred_char_tblro, pred_char_cls,
                true_char_fg, true_char_tblro, true_char_cls):
        # FG LOSS: DICE + BCE
        smooth = 1
        pred_char_fg = torch.reshape(pred_char_fg[:, 1], (-1,))
        true_char_fg = true_char_fg.view(-1)
        intersection = (pred_char_fg * true_char_fg).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(pred_char_fg.sum() + true_char_fg.sum() + smooth)
        bce = F.binary_cross_entropy(pred_char_fg, true_char_fg, reduction='mean')
        fg_loss = bce + dice_loss * 0.1

        # TBLRO LOSS: IOU + specific angle loss
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(true_char_tblro, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(pred_char_tblro, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        aabb_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        theta_loss = 1 - torch.cos(theta_pred - theta_gt)
        tblro_loss = aabb_loss + 10 * theta_loss

        # CHAR CLASSIFICATION LOSS: BCE
        cls_loss = F.binary_cross_entropy(pred_char_cls.view(-1), true_char_cls.view(-1), reduction='mean')

        return fg_loss + torch.mean(tblro_loss) + cls_loss
