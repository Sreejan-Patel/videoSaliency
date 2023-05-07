import torch
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image
import cv2 as cv

def torch_transform_image(img):
    transform = transforms.Compose([
            transforms.Resize((224, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
    ])
    img = transform(img)
    return img


def blur(img):
    k_size = 11
    blurred = cv.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(blurred)


def save_image(tensor, save_path,
               n_row=8, padding=2, pad_value=0, normalize=False, value_range=None, scale_each=False):
    grid = utils.make_grid(
        tensor,
        nrow=n_row,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each
    )

    nd_arr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    nd_arr = nd_arr[:, :, 0]
    img = Image.fromarray(nd_arr)
    extension = save_path.split('.')[-1]
    if extension == "png":
        img.save(save_path)
    else:
        img.save(save_path, quality=100)  # for jpg
