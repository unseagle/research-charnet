import torch
import torch.nn as nn
import torch.nn.functional as F

from util import weighted_sum


class CombinedLoss(nn.Module):
    def __init__(self, word_char=0.5, fg_tblro=0.5, aabb_theta=0.1, bce_dice=0.9):
        super(CombinedLoss, self).__init__()
        self.calc_np = False

        self.param_word_char = word_char
        self.param_fg_tblro = fg_tblro
        self.param_aabb_theta = aabb_theta
        self.param_bce_dice = bce_dice

    def calc_loss(self, pred_fg, pred_tblro, true_fg, true_tblro):
        if self.calc_np:
            # numpy error Ausgaben
            np_true_fg = true_fg.cpu().detach().numpy()
            np_pred_fg = pred_fg.cpu().detach().numpy()

        # FG LOSS: DICE + BCE
        smooth = 1
        pred_fg = torch.reshape(pred_fg[:, 1], (-1,))
        true_fg = true_fg.view(-1)
        intersection = (pred_fg * true_fg).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_fg.sum() + true_fg.sum() + smooth)  # in [0,1)
        bce = F.binary_cross_entropy(pred_fg, true_fg, reduction='mean')  # in [0,1)
        fg_loss = weighted_sum(bce, dice_loss, self.param_bce_dice)  # in [0,1.1)

        # TBLRO LOSS: IOU + specific angle loss
        dt_gt, db_gt, dl_gt, dr_gt, theta_gt = torch.split(true_tblro, 1, 1)
        dt_pred, db_pred, dl_pred, dr_pred, theta_pred = torch.split(pred_tblro, 1, 1)
        area_gt = (dt_gt + db_gt) * (dr_gt + dl_gt)
        area_pred = (dt_pred + db_pred) * (dr_pred + dl_pred)
        w_union = torch.min(dr_gt, dr_pred) + torch.min(dl_gt, dl_pred)
        h_union = torch.min(dt_gt, dt_pred) + torch.min(db_gt, db_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        aabb_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))  # + 1 probably because of div by zero
        theta_loss = 1 - torch.cos(theta_pred - theta_gt)
        tblro_loss = weighted_sum(aabb_loss, theta_loss, self.param_aabb_theta)

        tblro_mask = true_fg > 0.9
        tblro_mask = tblro_mask.reshape(tblro_loss.shape)

        tblro_loss_masked = tblro_loss[tblro_mask]

        tblro_loss_mean = torch.mean(tblro_loss[tblro_mask]) if len(tblro_loss_masked) > 0 else 0

        if self.calc_np:
            # numpy error Ausgaben
            inv_mask = true_fg <= 0.9
            inv_mask = inv_mask.reshape(tblro_loss.shape)
            np_dt_gt, np_db_gt, np_dl_gt, np_dr_gt, np_theta_gt = \
                dt_gt.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                db_gt.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                dl_gt.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                dr_gt.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                theta_gt.masked_fill(inv_mask, 0).cpu().detach().numpy()
            np_dt_pred, np_db_pred, np_dl_pred, np_dr_pred, np_theta_pred = \
                dt_pred.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                db_pred.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                dl_pred.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                dr_pred.masked_fill(inv_mask, 0).cpu().detach().numpy(), \
                theta_pred.masked_fill(inv_mask, 0).cpu().detach().numpy()
            np_area_gt, np_area_pred = area_gt.cpu().detach().numpy(), area_pred.cpu().detach().numpy()
            np_w_union, np_h_union = w_union.cpu().detach().numpy(), h_union.cpu().detach().numpy()
            np_area_intersect, np_area_union = area_intersect.cpu().detach().numpy(), area_union.cpu().detach().numpy()
            np_aabb_loss = aabb_loss.cpu().detach().numpy()
            np_theta_loss = theta_loss.cpu().detach().numpy()
            np_tblro_loss = tblro_loss.cpu().detach().numpy()
            np_tblro_loss_masked = tblro_loss.masked_fill(inv_mask, 0).cpu().detach().numpy()

        return weighted_sum(fg_loss, tblro_loss_mean, self.param_fg_tblro)

    def forward(self, pred_word_fg, pred_word_tblro, pred_char_fg, pred_char_tblro,
                true_word_fg, true_word_tblro, true_char_fg, true_char_tblro):
        word_loss = self.calc_loss(pred_word_fg, pred_word_tblro, true_word_fg, true_word_tblro)
        char_loss = self.calc_loss(pred_char_fg, pred_char_tblro, true_char_fg, true_char_tblro)

        return weighted_sum(word_loss, char_loss, self.param_word_char)
