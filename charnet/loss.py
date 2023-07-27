import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, pred_char_fg, pred_char_tblro, true_char_fg, true_char_tblro):
        # FG LOSS: DICE + BCE
        smooth = 1
        pred_char_fg = torch.reshape(pred_char_fg[:, 1], (-1,))
        true_char_fg = true_char_fg.view(-1)
        intersection = (pred_char_fg * true_char_fg).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(pred_char_fg.sum() + true_char_fg.sum() + smooth)  # in [0,1)
        bce = F.binary_cross_entropy(pred_char_fg, true_char_fg, reduction='mean')  # in [0,1)
        fg_loss = bce + dice_loss * 0.1  # in [0,1.1)

        # TBLRO LOSS: IOU + specific angle loss
        dt_gt, db_gt, dl_gt, dr_gt, theta_gt = torch.split(true_char_tblro, 1, 1)
        # dt_pred, db_pred, dl_pred, dr_pred, theta_pred = torch.split(pred_char_tblro, 1, 1)
        dt_pred, db_pred, dl_pred, dr_pred = torch.split(pred_char_tblro, 1, 1)
        area_gt = (dt_gt + db_gt) * (dr_gt + dl_gt)
        area_pred = (dt_pred + db_pred) * (dr_pred + dl_pred)
        w_union = torch.min(dr_gt, dr_pred) + torch.min(dl_gt, dr_pred)
        h_union = torch.min(dt_gt, dt_pred) + torch.min(db_gt, db_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        aabb_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))  # + 1 probably because of div by zero
        # theta_loss = 1 - torch.cos(theta_pred - theta_gt)
        # tblro_loss = aabb_loss + 10 * theta_loss
        tblro_loss = aabb_loss  # in [0,infinity)

        return fg_loss + torch.mean(tblro_loss)  # in [0,infinity)
