import torch.nn as nn
import torch


class OptunaLoss(nn.Module):
    def __init__(self, weight=0.5):
        super(OptunaLoss, self).__init__()
        self.weight = weight

    def iou(self, pred, gt):
        # pred_np = pred.cpu().detach().numpy()
        # gt_np = gt.cpu().detach().numpy()
        pred = (pred > 0.5).type(torch.FloatTensor)
        gt = (gt > 0.5).type(torch.FloatTensor)
        # pred_np = pred.cpu().detach().numpy()
        # gt_np = gt.cpu().detach().numpy()
        i = torch.sum(pred * gt)
        u = torch.sum(pred) + torch.sum(gt) - i

        iou_loss = 1.0 - (i + 0.0001) / (u + 0.0001)
        return iou_loss.item()

    def forward(self, char_fg_pred, char_fg_gt):
        return self.weight * self.iou(char_fg_pred[:, 0], char_fg_gt[:, 0]) \
            + (1 - self.weight) * self.iou(1 - char_fg_pred[:, 0], 1 - char_fg_gt[:, 0])
