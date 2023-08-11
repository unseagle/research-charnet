import math

import cv2 as cv
import numpy as np
import torch

import config
from datetime import datetime


def draw_bbs(imgs, bbs, is_normalized=False):
    result_imgs = []
    for i, img in enumerate(imgs):
        if not is_normalized:
            img = img * 255.
        if len(bbs) > i:
            cur_bbs = bbs[i]
            for bb in cur_bbs:
                cv.polylines(img, [bb[:8].reshape((-1, 2)).astype(np.int32)], True,
                             (0., 255, 0.), 1)
        result_imgs.append(img)
    return np.array(result_imgs)


def transpose_for_cv(imgs):
    return np.array([img.transpose(1, 2, 0) for img in imgs])


def weighted_sum(a, b, weight_a):
    return weight_a * a + (1 - weight_a) * b


def get_branch_list():
    return [
        config.ACTIVATE_CHAR_FG_BCE or config.ACTIVATE_WORD_FG_DICE,
        config.ACTIVATE_WORD_TBLR, config.ACTIVATE_WORD_ORIENT,
        config.ACTIVATE_CHAR_FG_BCE or config.ACTIVATE_CHAR_FG_DICE,
        config.ACTIVATE_CHAR_TBLR, config.ACTIVATE_CHAR_ORIENT
    ]


def save_weights(model, optimizer, epoch):
    timestr = datetime.now().strftime("%m-%d_%H-%M")
    branches = "".join(map(str, list(map(lambda x: 1 if x else 0, get_branch_list()))))
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'branches': branches
    }
    torch.save(state, f"{config.weights_folder}{timestr}_branches_{branches}.pth")


# visualizations

def visualize_batch(model, data_loader, prefix, print_unchanged=False):
    name_prefix = config.print_batch_folder + prefix
    w_fg_bce = config.ACTIVATE_WORD_FG_BCE
    w_fg_dice = config.ACTIVATE_WORD_FG_DICE
    w_tblr = config.ACTIVATE_WORD_TBLR
    w_orient = config.ACTIVATE_WORD_ORIENT
    c_fg_bce = config.ACTIVATE_CHAR_FG_BCE
    c_fg_dice = config.ACTIVATE_CHAR_FG_DICE
    c_tblr = config.ACTIVATE_CHAR_TBLR
    c_orient = config.ACTIVATE_CHAR_ORIENT
    w_fg = w_fg_bce or w_fg_dice
    c_fg = c_fg_bce or c_fg_dice
    w = w_fg and w_tblr and w_orient
    c = c_fg and c_tblr and c_orient

    model.eval()
    for _, (data, targets) in enumerate(data_loader):
        model.return_all = True
        char_bbs, word_bbs, word_fg, word_tblro, char_fg, char_tblro = model(data[0:config.print_batch_num])
        model.return_all = False
        imgs = data[0:config.print_batch_num].cpu().detach().numpy() * 255.
        imgs = transpose_for_cv(imgs)
        if print_unchanged:
            visualize_img_batch(imgs, config.print_batch_folder + "0riginal_", is_normalized=True)
        if w:
            imgs = draw_bbs(imgs, word_bbs, is_normalized=True)
        if c:
            imgs = draw_bbs(imgs, char_bbs, is_normalized=True)
        if w or c:
            visualize_img_batch(imgs, name_prefix + "_0_bboxes", is_normalized=True)
        if not w:
            if w_fg:
                visualize_img_batch(word_fg.cpu().detach().numpy()[:, 1], name_prefix + "_word_fg", is_normalized=False)
        if not c:
            if c_fg:
                visualize_img_batch(char_fg.cpu().detach().numpy()[:, 1], name_prefix + "_char_fg", is_normalized=False)
        if not w and w_tblr:
            # tblr_map = (targets[1].cpu().detach().numpy()[0:config.print_batch_num, 0:4])
            tblr_map = (word_tblro.cpu().detach().numpy()[:, 0:4])
            for current_dir, current_map in zip(["top", "bottom", "left", "right"],
                                                np.transpose(tblr_map, (1, 0, 2, 3))):
                current_map = scale_to_0_255(mask_img(current_map, targets[0][0:config.print_batch_num]))
                heat_maps = []
                for idx, single_map in enumerate(current_map):
                    heat_maps.append(cv.applyColorMap(single_map, cv.COLORMAP_JET))
                heat_maps = np.array(heat_maps)
                mask_rgb(heat_maps, targets[0][0:config.print_batch_num].cpu().detach().numpy())
                visualize_img_batch(heat_maps, name_prefix + f"_word_dir_{current_dir}", is_normalized=True)
        if not c and c_tblr:
            tblr_map = (char_tblro.cpu().detach().numpy()[:, 0:4])
            for current_dir, current_map in zip(["top", "bottom", "left", "right"],
                                                np.transpose(tblr_map, (1, 0, 2, 3))):
                current_map = scale_to_0_255(mask_img(current_map, targets[0][0:config.print_batch_num]))
                heat_maps = []
                for idx, single_map in enumerate(current_map):
                    heat_maps.append(cv.applyColorMap(single_map, cv.COLORMAP_JET))
                visualize_img_batch(heat_maps, name_prefix + f"_char_dir_{current_dir}", is_normalized=True)
        if not w and w_orient:
            # o_map = (targets[1][0:config.print_batch_num].cpu().detach().numpy()[:, 4]) * (127.5 / (math.pi / 2 + 0.1)) + 127.5
            o_map = (word_tblro.cpu().detach().numpy()[:, 4]) * (127.5 / (math.pi / 2 + 0.1)) + 127.5
            heat_maps = []
            for o in o_map:
                heat_maps.append(cv.applyColorMap(o.astype(np.uint8), cv.COLORMAP_JET))

            heat_maps = np.array(heat_maps)
            mask_rgb(heat_maps, targets[0][0:config.print_batch_num].cpu().detach().numpy())
            visualize_img_batch(heat_maps, name_prefix + "_word_orient", is_normalized=True)
        if not c and c_orient:
            # o_map = (targets[3][0:config.print_batch_num].cpu().detach().numpy()[:, 4]) * (127.5 / (math.pi / 2 + 0.1)) + 127.5
            o_map = (char_tblro.cpu().detach().numpy()[:, 4]) * (127.5 / (math.pi / 2 + 0.1)) + 127.5
            heat_maps = []
            for o in o_map:
                heat_maps.append(cv.applyColorMap(o.astype(np.uint8), cv.COLORMAP_JET))

            heat_maps = np.array(heat_maps)
            mask_rgb(heat_maps, targets[2][0:config.print_batch_num].cpu().detach().numpy())
            visualize_img_batch(heat_maps, name_prefix + "_char_orient", is_normalized=True)

        break


def visualize_img_batch(img_batch, file, is_normalized=False):
    for idx, img in enumerate(img_batch):
        if not is_normalized:
            img = img * 255.0
        cv.imwrite(file + f"_{idx}.png", img)


def mask_img(img, fg):
    mask = fg < 0.1
    mask = mask.reshape(img.shape)
    img = img.copy()
    img[mask] = 0
    return img


# expects fg in shape (batches, 1, dim, dim)
def mask_rgb(imgs, fg, val=0):
    mask = (lambda x: np.array([x, x, x]))(fg).transpose((2, 1, 3, 4, 0))[0]
    mask = (mask < 0.1)
    imgs[mask] = val


def scale_to_0_255(img_batch):
    res = []
    for img in img_batch:
        factor = 255 / (np.max(img) if np.max(img) > 0 else 1)
        res.append(img * factor)
    return np.array(res).astype(np.uint8)
