import cv2 as cv
import numpy as np


def draw_bbs(imgs, word_bbs, char_bbs):
    result_imgs = []
    for i in range(len(imgs)):
        img = (imgs[i] * 255.).cpu().numpy().transpose(1, 2, 0).copy()
        if len(word_bbs) > i:
            cur_word_bbs = word_bbs[i]
            for word_bb in cur_word_bbs:
                cv.polylines(img, [word_bb[:8].reshape((-1, 2)).astype(np.int32)], True,
                             (0., 255, 0.), 1)
        if len(char_bbs) > i:
            cur_char_bbs = char_bbs[i]
            for char_bb in cur_char_bbs:
                cv.polylines(img, [char_bb[:8].reshape((-1, 2)).astype(np.int32)], True,
                             (0., 255, 0.), 1)
        result_imgs.append(img)
    return result_imgs
