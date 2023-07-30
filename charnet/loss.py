import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

    def calc_loss(self, pred_fg, pred_tblro, true_fg, true_tblro):
        # FG LOSS: DICE + BCE
        smooth = 1
        pred_fg = torch.reshape(pred_fg[:, 1], (-1,))
        true_fg = true_fg.view(-1)
        intersection = (pred_fg * true_fg).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_fg.sum() + true_fg.sum() + smooth)  # in [0,1)
        bce = F.binary_cross_entropy(pred_fg, true_fg, reduction='mean')  # in [0,1)
        fg_loss = bce + dice_loss * 0.1  # in [0,1.1)

        # TBLRO LOSS: IOU + specific angle loss
        dt_gt, db_gt, dl_gt, dr_gt, theta_gt = torch.split(true_tblro, 1, 1)
        dt_pred, db_pred, dl_pred, dr_pred, theta_pred = torch.split(pred_tblro, 1, 1)
        # dt_pred, db_pred, dl_pred, dr_pred = torch.split(pred_tblro, 1, 1)
        area_gt = (dt_gt + db_gt) * (dr_gt + dl_gt)
        area_pred = (dt_pred + db_pred) * (dr_pred + dl_pred)
        w_union = torch.min(dr_gt, dr_pred) + torch.min(dl_gt, dr_pred)
        h_union = torch.min(dt_gt, dt_pred) + torch.min(db_gt, db_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        aabb_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))  # + 1 probably because of div by zero
        theta_loss = 1 - torch.cos(theta_pred - theta_gt)
        tblro_loss = aabb_loss + 10 * theta_loss  # in [0,infinity) plus whatever theta_loss can give
        # tblro_loss = aabb_loss  # in [0,infinity)

        return fg_loss + torch.mean(tblro_loss)  # in [0,infinity)

    def forward(self, pred_word_fg, pred_word_tblro, pred_char_fg, pred_char_tblro,
                true_word_fg, true_word_tblro, true_char_fg, true_char_tblro):
        word_loss = self.calc_loss(pred_word_fg, pred_word_tblro, true_word_fg, true_word_tblro)
        char_loss = self.calc_loss(pred_char_fg, pred_char_tblro, true_char_fg, true_char_tblro)

        word_part = 0.5

        return word_part * word_loss + (1 - word_part) * char_loss
