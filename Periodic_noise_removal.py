import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from params import max_size
from PIL import Image


def get_vgg19_model():
    """
    Extract the CNN layers of VGG19
    """
    vgg_cnn_model = models.vgg19(pretrained=True).features
    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg_cnn_model.parameters():
        param.requires_grad_(False)
    return vgg_cnn_model


def get_device():
    """
    Checks for GPU availability.
    :returns use_gpu : Flag indication gpu is available
    :returns device : The torch device
    """
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("gpu")
        print("Running on GPU!!!")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
    return use_gpu, device


def load_image_as_tensor(img_path, shape=None):
    """
    Load time from path and convert to a tensor. Images will be resized to maximum of max_limit
    :param img_path = Image Path
    :param shape = input shape to resize
    :returns image tensor
    """
    image = Image.open(img_path).convert('RGB')
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    im_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = im_transform(image)[:3, :, :].unsqueeze(0)

    return image
