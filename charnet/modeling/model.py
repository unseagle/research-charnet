# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from charnet.modeling.backbone.hourglass import hourglass88
from collections import OrderedDict
from torch.functional import F
import torchvision.transforms as T
from .postprocessing import OrientedTextPostProcessing
from charnet.config import cfg
from ..loss import CombinedLoss


def _conv3x3_bn_relu(in_channels, out_channels, dilation=1):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False
        )),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.ReLU())
    ]))


def to_numpy_or_none(*tensors):
    results = []
    for t in tensors:
        if t is None:
            results.append(None)
        else:
            results.append(t.cpu().detach().numpy())
    return results


class WordDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, dilation=1):
        super(WordDetector, self).__init__()
        self.word_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels, dilation
        )
        self.word_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.word_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.word_det_conv_final(x)

        pred_word_fg = self.word_fg_pred(self.word_fg_feat(feat))

        word_regression_feat = self.word_regression_feat(feat)
        pred_word_tblr = F.relu(self.word_tblr_pred(word_regression_feat)) * 10.
        pred_word_orient = self.orient_pred(word_regression_feat)

        return pred_word_fg, pred_word_tblr, pred_word_orient


class CharDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, curved_text_on=False):
        super(CharDetector, self).__init__()
        self.character_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels
        )
        self.char_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.char_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.character_det_conv_final(x)

        pred_char_fg = self.char_fg_pred(self.char_fg_feat(feat))
        char_regression_feat = self.char_regression_feat(feat)
        pred_char_tblr = F.relu(self.char_tblr_pred(char_regression_feat)) * 10.
        pred_char_orient = self.orient_pred(char_regression_feat)

        return pred_char_fg, pred_char_tblr, pred_char_orient


class CharRecognizer(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, num_classes):
        super(CharRecognizer, self).__init__()

        self.body = nn.Sequential(
            _conv3x3_bn_relu(in_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
        )
        self.classifier = nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        feat = self.body(feat)
        return self.classifier(feat)


class CharNet(nn.Module):
    def __init__(self, backbone=hourglass88(), img_size=256):
        super(CharNet, self).__init__()
        self.backbone = backbone
        decoder_channels = 256
        bottleneck_channels = 128
        self.word_detector = WordDetector(
            decoder_channels, bottleneck_channels,
            dilation=cfg.WORD_DETECTOR_DILATION
        )
        self.char_detector = CharDetector(
            decoder_channels,
            bottleneck_channels
        )

        args = {
            "word_min_score": cfg.WORD_MIN_SCORE,
            "word_stride": cfg.WORD_STRIDE,
            "word_nms_iou_thresh": cfg.WORD_NMS_IOU_THRESH,
            "char_stride": cfg.CHAR_STRIDE,
            "char_min_score": cfg.CHAR_MIN_SCORE,
            "num_char_class": cfg.NUM_CHAR_CLASSES,
            "char_nms_iou_thresh": cfg.CHAR_NMS_IOU_THRESH,
            "char_dict_file": cfg.CHAR_DICT_FILE,
            "word_lexicon_path": cfg.WORD_LEXICON_PATH
        }

        self.post_processing = OrientedTextPostProcessing(**args)

        self.transform = self.build_transform()

        self.return_bbs = False
        self.return_all = False
        self.img_size = img_size

    # def forward(self, im, im_scale_w, im_scale_h, original_im_w, original_im_h):
    def forward(self, im: torch.Tensor):
        im = torch.stack([self.transform(oneimg).cuda() for oneimg in im])
        # im = im.unsqueeze(0)
        features = self.backbone(im)

        pred_word_fg, pred_word_tblr, pred_word_orient = self.word_detector(features)
        pred_char_fg, pred_char_tblr, pred_char_orient = self.char_detector(features)

        pred_word_fg = F.softmax(pred_word_fg, dim=1)
        pred_char_fg = F.softmax(pred_char_fg, dim=1)

        if not self.return_all and not self.return_bbs:
            pred_word_tblro = torch.cat((pred_word_tblr, pred_word_orient), 1)
            pred_char_tblro = torch.cat((pred_char_tblr, pred_char_orient), 1)

            return pred_word_fg, pred_word_tblro, pred_char_fg, pred_char_tblro

        pred_word_fg_np, pred_word_tblr_np, \
            pred_word_orient_np, pred_char_fg_np, \
            pred_char_tblr_np, pred_char_orient_np = to_numpy_or_none(
            pred_word_fg, pred_word_tblr,
            pred_word_orient, pred_char_fg,
            pred_char_tblr, pred_char_orient
        )

        # char_bboxes, word_instances = self.post_processing(
        #     pred_word_fg[0, 1], pred_word_tblr[0],
        #     pred_word_orient[0, 0], pred_char_fg[0, 1],
        #     pred_char_tblr[0],
        #     1, 1, self.img_size, self.img_size
        # )

        char_bboxes = []
        word_bboxes = []
        for i in range(len(pred_word_fg_np)):
            char_bbox, word_instance = self.post_processing(
                pred_word_fg_np[i, 1], pred_word_tblr_np[i],
                pred_word_orient_np[i, 0], pred_char_fg_np[i, 1],
                pred_char_tblr_np[i], 1, 1, self.img_size, self.img_size
            )
            char_bboxes.append(char_bbox)
            word_bboxes.append(word_instance)

        if self.return_all:
            pred_word_tblro = torch.cat((pred_word_tblr, pred_word_orient), 1)
            pred_char_tblro = torch.cat((pred_char_tblr, pred_char_orient), 1)
            return char_bboxes, word_bboxes, pred_word_fg, pred_word_tblro, pred_char_fg, pred_char_tblro
        return char_bboxes, word_bboxes

    def build_transform(self):
        to_rgb_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_rgb_transform,
                normalize_transform,
            ]
        )
        return transform
