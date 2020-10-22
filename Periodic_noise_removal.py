import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from params import max_size, path_name_noise
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


def im_convert(tensor):
    """ Extract image from tensor
    :param tensor: input torch tensor
     """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def get_features(image, model, layers=None):
    """ Extract feature maps form vgg network layers
    :param image : input image
    :param model : vgg model
    :param layers : Layers from which feature need to be collected
    """
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1'}

    feature_maps = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            feature_maps[layers[name]] = x

    return feature_maps


def find_gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    _, depth, height, width = tensor.size()
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    return gram


def generate_random_image():
    image_array = np.random.rand(100,100,3) * 255
    im = Image.fromarray(image_array.astype('uint8')).convert('RGBA')
    im.save(path_name_noise)

def train_image(model, optimizer, steps):
progress = 0
for ii in range(1, steps + 1):
    progress += 1
    # get the features from your target image
    target_features = get_features(target, vgg)

    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % show_every == 0:
        sys.stdout.write("\rProgress:" +
                         str(100 * progress / float(ii))[:4] \
                         + "%" + " Total loss:" + str((total_loss.item()))[0:5] \
                         + " Style Transfer ")
        plt.imshow(im_convert(target))
        plt.show()
