import torch
import os
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
import math
import torchvision.transforms as transforms
from torch.utils import data
import charnet.config


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    # current_angle = math.atan((vertices[3] - vertices[1]) / (vertices[2] - vertices[0] + 0.00001))
    # if vertices[2] - vertices[0] < 0:
    #     current_angle += math.pi  # * (-1 if current_angle > 0 else 1)
    # if current_angle < 0:
    #     current_angle += 2 * math.pi
    #
    # if math.pi / 4 >= current_angle:
    #     return -1 * current_angle
    # if current_angle > math.pi * 7 / 4:
    #     return current_angle - 2 * math.pi
    # elif math.pi * 3 / 4 >= current_angle > math.pi / 4:
    #     return (math.pi / 2) - current_angle
    # elif math.pi * 5 / 4 >= current_angle > math.pi * 3 / 4:
    #     return math.pi - current_angle
    # else:
    #     return (math.pi * 3 / 2) - current_angle
    #
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


# def is_cross_text(start_loc, length, vertices):
#     '''check if the crop image crosses text regions
#     Input:
#         start_loc: left-top position
#         length   : length of crop image
#         vertices : vertices of text regions <numpy.ndarray, (n,8)>
#     Output:
#         True if crop image crosses text region
#     '''
#     if vertices.size == 0:
#         return False
#     start_w, start_h = start_loc
#     a = np.array([start_w, start_h, start_w + length, start_h,
#                   start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
#     p1 = Polygon(a).convex_hull
#     for vertice in vertices:
#         p2 = Polygon(vertice.reshape((4, 2))).convex_hull
#         inter = p1.intersection(p2).area
#         if 0.01 <= inter / p2.area <= 0.99:
#             return True
#     return False


# def crop_img(img, vertices, labels, length):
#     '''crop img patches to obtain batch and augment
#     Input:
#         img         : PIL Image
#         vertices    : vertices of text regions <numpy.ndarray, (n,8)>
#         labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
#         length      : length of cropped image region
#     Output:
#         region      : cropped image region
#         new_vertices: new vertices in cropped region
#     '''
#     h, w = img.height, img.width
#     # confirm the shortest side of image >= length
#     if h >= w and w < length:
#         img = img.resize((length, int(h * length / w)), Image.BILINEAR)
#     elif h < w and h < length:
#         img = img.resize((int(w * length / h), length), Image.BILINEAR)
#     ratio_w = img.width / w
#     ratio_h = img.height / h
#     assert (ratio_w >= 1 and ratio_h >= 1)
#
#     new_vertices = np.zeros(vertices.shape)
#     if vertices.size > 0:
#         new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
#         new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h
#
#     # find random position
#     remain_h = img.height - length
#     remain_w = img.width - length
#     flag = True
#     cnt = 0
#     while flag and cnt < 1000:
#         cnt += 1
#         start_w = int(np.random.rand() * remain_w)
#         start_h = int(np.random.rand() * remain_h)
#         flag = is_cross_text([start_w, start_h], length, new_vertices[labels == 1, :])
#     box = (start_w, start_h, start_w + length, start_h + length)
#     region = img.crop(box)
#     if new_vertices.size == 0:
#         return region, new_vertices
#
#     new_vertices[:, [0, 2, 4, 6]] -= start_w
#     new_vertices[:, [1, 3, 5, 7]] -= start_h
#     return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                    np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


# def adjust_height(img, vertices, ratio=0.2):
#     '''adjust height of image to aug data
#     Input:
#         img         : PIL Image
#         vertices    : vertices of text regions <numpy.ndarray, (n,8)>
#         ratio       : height changes in [0.8, 1.2]
#     Output:
#         img         : adjusted PIL Image
#         new_vertices: adjusted vertices
#     '''
#     ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
#     old_h = img.height
#     new_h = int(np.around(old_h * ratio_h))
#     img = img.resize((img.width, new_h), Image.BILINEAR)
#
#     new_vertices = vertices.copy()
#     if vertices.size > 0:
#         new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
#     return img, new_vertices


# def rotate_img(img, vertices, angle_range=10):
#     '''rotate image [-10, 10] degree to aug data
#     Input:
#         img         : PIL Image
#         vertices    : vertices of text regions <numpy.ndarray, (n,8)>
#         angle_range : rotate range
#     Output:
#         img         : rotated PIL Image
#         new_vertices: rotated vertices
#     '''
#     center_x = (img.width - 1) / 2
#     center_y = (img.height - 1) / 2
#     angle = angle_range * (np.random.rand() * 2 - 1)
#     img = img.rotate(angle, Image.BILINEAR)
#     new_vertices = np.zeros(vertices.shape)
#     for i, vertice in enumerate(vertices):
#         new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
#     return img, new_vertices


def get_score_geo(img, vertices, labels, word_vertices, scale, length):
    '''generate score gt and geometry gt
    Input:
        img     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image
        length  : image length
    Output:
        score gt, geo gt, ignored
    '''
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    cls_map = np.zeros((int(img.height * scale), int(img.width * scale), charnet.config.cfg.NUM_CHAR_CLASSES), np.float32)

    score_map_word = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map_word = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    polys = []

    for i, vertex in enumerate(vertices):
        poly = np.around(scale * shrink_poly(vertex).reshape((4, 2))).astype(np.int32)  # scaled & shrunk
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertex)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertex, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertex[0], vertex[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

        cv.fillPoly(np.int32(cls_map[:, :, labels[i]]), [poly], 1)
    cv.fillPoly(score_map, polys, 1)

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    polys = []

    for i, vertex in enumerate(word_vertices):
        poly = np.around(scale * shrink_poly(vertex).reshape((4, 2))).astype(np.int32)  # scaled & shrunk
        polys.append(poly)
        temp_mask = np.zeros(score_map_word.shape[:-1], np.float32)
        cv.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertex)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertex, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertex[0], vertex[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map_word[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map_word[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map_word[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map_word[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map_word[:, :, 4] += theta * temp_mask

    cv.fillPoly(score_map_word, polys, 1)
    return (torch.Tensor(score_map_word).permute(2, 0, 1), torch.Tensor(geo_map_word).permute(2, 0, 1),
            torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1),
            torch.Tensor(cls_map).permute(2, 0, 1))


def extract_vertices(lines, dictionary):
    '''extract vertices info from txt lines
    Input:
        lines   : list of string info
        dictionary
    Output:
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels: characters as number, translated via dictionary
    '''
    labels = []
    vertices = []
    word_vertices = []
    first = True
    actual_word = ""
    build_word = ""
    for line in lines:
        # first line in block is word line
        if line == "\n":
            assert actual_word == build_word
            first = True
            continue
        elif first:
            first_line = line.rstrip('\n').lstrip('\ufeff').split(',')
            word_vertices.append(list(map(int, first_line[:8])))
            actual_word = first_line[8]
            build_word = ""
            first = False
        else:
            orig_line = line.rstrip('\n')
            line = line.rstrip('\n').lstrip('\ufeff').split(',')
            vertices.append(list(map(int, line[:8])))
            character = line[8].upper()
            build_word += line[8]
            if character not in dictionary:
                if ",," == orig_line[-2:]:
                    labels.append(dictionary[','])
                else:
                    print(character, 'not in dictionary.')
                    labels.append(0)
            else:
                labels.append(dictionary[character])
            # sanity check, is the word correct
    return np.array(word_vertices), np.array(vertices), np.array(labels)


def load_char_dict(path, separator=chr(31)):
    char_dict = dict()
    with open(path, 'rt') as fr:
        for line in fr:
            sp = line.strip('\n').split(separator)
            char_dict[sp[0].upper()] = int(sp[1])
    return char_dict


class CustomDataset(data.Dataset):
    def __init__(self, samples, img_dir, mask_dir, scale=0.25, length=512):
        super(CustomDataset, self).__init__()
        self.length = length
        self.dictionary = load_char_dict(charnet.config.cfg.CHAR_DICT_FILE)

        self.img_labels = samples
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.scale = scale

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        mask_path = str(os.path.join(self.mask_dir, '.'.join(self.img_labels[idx].split('.')[:-1]))) + '.txt'

        with open(mask_path, 'r') as f:
            lines = f.readlines()

        word_vertices, vertices, labels = extract_vertices(lines, self.dictionary)

        img = Image.open(img_path)
        # img, vertices = adjust_height(img, vertices)
        # img, vertices = rotate_img(img, vertices)
        # img, vertices = crop_img(img, vertices, labels, self.length)
        transform = transforms.Compose([  # transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
            transforms.Normalize(mean=0.5, std=0.5)])

        if img.height != self.length:
            print(f"Wrong dimensions, img.height={img.height}, img.width={img.width}")
            print(img_path)
        true_word_fg, true_word_tblro, true_char_fg, true_char_tblro, true_char_cls = get_score_geo(img, vertices, labels, word_vertices, self.scale, self.length)
        img = np.array(img)
        img = np.stack((img,) * 3, axis=2)  # RGB
        img = img / 255.0  # (512, 512)
        img = np.transpose(img, (2, 0, 1))   # (3, 512, 512)
        img = img.astype(np.float32)
        img = torch.from_numpy(img)

        return transform(img), (true_word_fg, true_word_tblro, true_char_fg, true_char_tblro)
