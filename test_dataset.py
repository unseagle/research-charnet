import torch

from charnet.dataset import CustomDataset

dataset = CustomDataset(["ZB_0125_p1_PHBFDOC.jpg"], "example_samples/images", "example_samples/labels")
obj, gt = next(iter(dataset))
np_fg = gt[0].numpy()
nparr = obj.numpy()
print(len(dataset))
print(torch.count_nonzero(gt[0]))
pass
