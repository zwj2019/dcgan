import argparse
import os

import torch
from PIL import Image
import numpy as np

from model.dcgan import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--weights', help="Path to the checkpoints of the Generator", required=True, type=str)
parser.add_argument('--samples', help="Path to the sample images", default='samples', type=str)
parser.add_argument('--num', help="Number of sample images to generate", default=16, type=int)
parser.add_argument('--ngpu', help="Number of GPU, 0 for CPU", type=int, default=1)

args = parser.parse_args()

device = 'cuda' if args.ngpu > 0 else 'cpu'
if not os.path.exists(args.samples):
    os.mkdir(args.samples)

model = DCGAN.load_from_checkpoint(args.weights).to(device)

noise = torch.randn(args.num, 100, 1, 1, device=device)
fake_image = model(noise).detach().cpu()

img_tensor_list = [tensor for tensor in fake_image]
# tensor to narray
img_list = [img.numpy().transpose((1, 2, 0)) for img in img_tensor_list]

for i in range(len(img_list)):
    # Inverse normalization
    img = (img_list[i] * 0.5 + 0.5) * 255
    img = Image.fromarray(np.uint8(img))
    img.save(os.path.join(args.samples, '%d.jpg' % i))
