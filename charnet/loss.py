import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from util import weighted_sum


class CombinedLoss(nn.Module):
    def __init__(self, word_char=0.5, fg_tblro=0.5, aabb_theta=0.1, bce_dice=0.9):
        super(CombinedLoss, self).__init__()
        self.calc_np = False

        self.param_word_char = word_char
        self.param_fg_tblro = fg_tblro
        self.param_aabb_theta = aabb_theta
        self.param_bce_dice = bce_dice

    def calc_loss(self, pred_fg, pred_tblro, true_fg, true_tblro,
                  act_fg_bce, act_fg_dice, act_tblr, act_orient):
        if not (act_fg_bce or act_fg_dice or act_tblr or act_orient):
            return 0
        # FG LOSS: DICE + BCE
        smooth = 1
        pred_fg = torch.reshape(pred_fg[:, 1], (-1,))
        orig_true_fg = true_fg.detach().clone()
        true_fg = true_fg.view(-1)

        dice_loss = 0
        bce = 0
        if act_fg_dice:
            intersection = (pred_fg * true_fg).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (pred_fg.sum() + true_fg.sum() + smooth)  # in [0,1)
        if act_fg_bce:
            bce = F.binary_cross_entropy(pred_fg, true_fg, reduction='mean')  # in [0,1)

        fg_loss = weighted_sum(bce, dice_loss, self.param_bce_dice)  # in [0,1.1)

        # TBLRO LOSS: IOU + specific angle loss
        dt_gt, db_gt, dl_gt, dr_gt, theta_gt = torch.split(true_tblro, 1, 1)
        dt_pred, db_pred, dl_pred, dr_pred, theta_pred = torch.split(pred_tblro, 1, 1)
        tblro_mask = orig_true_fg > 0.9
        # tblro_mask = tblro_mask.reshape(true_fg.shape)[:, 0]

        tblr_loss = 0
        if act_tblr:
            area_gt = (dt_gt + db_gt) * (dr_gt + dl_gt)
            area_pred = (dt_pred + db_pred) * (dr_pred + dl_pred)
            w_union = torch.min(dr_gt, dr_pred) + torch.min(dl_gt, dl_pred)
            h_union = torch.min(dt_gt, dt_pred) + torch.min(db_gt, db_pred)
            area_intersect = w_union * h_union
            area_union = area_gt + area_pred - area_intersect
            tblr_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))  # + 1 probably because of div by zero
            tblr_loss = tblr_loss[tblro_mask]
            tblr_loss = torch.mean(tblr_loss) if len(tblr_loss) > 0 else 0
        theta_loss = 0
        if act_orient:
            theta_loss = 1 - torch.cos(theta_pred - theta_gt)
            theta_loss = theta_loss[tblro_mask]
            theta_loss = torch.mean(theta_loss) if len(theta_loss) > 0 else 0

        tblro_loss = weighted_sum(tblr_loss, theta_loss, self.param_aabb_theta)

        loss = weighted_sum(fg_loss, tblro_loss, self.param_fg_tblro)

        return loss

    def forward(self, pred_word_fg, pred_word_tblro, pred_char_fg, pred_char_tblro,
                true_word_fg, true_word_tblro, true_char_fg, true_char_tblro):
        word_loss = self.calc_loss(pred_word_fg, pred_word_tblro, true_word_fg, true_word_tblro,
                                   config.ACTIVATE_WORD_FG_BCE,
                                   config.ACTIVATE_WORD_FG_DICE,
                                   config.ACTIVATE_WORD_TBLR,
                                   config.ACTIVATE_WORD_ORIENT)
        char_loss = self.calc_loss(pred_char_fg, pred_char_tblro, true_char_fg, true_char_tblro,
                                   config.ACTIVATE_CHAR_FG_BCE,
                                   config.ACTIVATE_CHAR_FG_DICE,
                                   config.ACTIVATE_CHAR_TBLR,
                                   config.ACTIVATE_CHAR_ORIENT)

        return weighted_sum(word_loss, char_loss, self.param_word_char)
