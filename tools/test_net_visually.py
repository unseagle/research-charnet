import torch
import cv2 as cv
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from charnet.modeling.model import CharNet
from charnet.config import cfg

def get_img(path):
    transform = transforms.Compose([transforms.Normalize(mean=0.5, std=0.5)])
    img = Image.open(path)
    img = np.array(img)
    img = np.stack((img,) * 3, axis=2)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img = torch.from_numpy(img)

    return transform(img).unsqueeze(0)


cfg.merge_from_file("configs/myconfig.yaml")
cfg.freeze()

model = CharNet()
model.to("cuda")

# decide what weights to load
weights = "myweights/07-28_19-43.pth"

model.load_state_dict(torch.load(weights), strict=False)
model.eval()
model.return_bbs = True

files = [f"example_samples/images/{f}" for f in os.listdir("example_samples/images")]

for path in files:
    print(f"Reading file {path}")
    img = get_img(path)
    char_bbs, word_bbs = model(img)
    print(f"char bbs: {char_bbs}")
    print(f"word bbs: {word_bbs}")
    for word_bb in word_bbs:
        word_bb_switched = np.zeros(8, dtype=int)
        cv.polylines(img[0].numpy().transpose(1, 2, 0), [word_bb[:8].reshape((-1, 2)).astype(np.int32)], True, (0, 255, 0), 1)
    for char_bb in char_bbs:
        char_bb_switched = np.zeros(8, dtype=int)
        cv.polylines(img[0].numpy().transpose(1, 2, 0), [char_bb[:8].reshape((-1, 2)).astype(np.int32)], True, (0, 255, 0),
                 1)
    cv.imshow("thingy", img[0].numpy().transpose(1, 2, 0))
    cv.waitKey(0)
    cv.destroyAllWindows()
