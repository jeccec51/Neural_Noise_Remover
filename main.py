import Periodic_noise_removal
from params import path_name_in, path_name_noise, layer_weights
import matplotlib.pyplot as plt
import torch.optim as optima
import params

nw_model = Periodic_noise_removal.get_vgg19_model()
sys_device = Periodic_noise_removal.get_device()
nw_model.to(sys_device)
Periodic_noise_removal.generate_random_image()
in_image = Periodic_noise_removal.load_image_as_tensor(path_name_in).to(sys_device)
rand_image = Periodic_noise_removal.load_image_as_tensor(path_name_noise, shape=in_image.shape[-2:]).to(sys_device)

# display the images
figure, (axis1, axis2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
axis1.imshow(Periodic_noise_removal.im_convert(in_image))
axis2.imshow(Periodic_noise_removal.im_convert(rand_image))
# get content and style features only once before training
image_features = Periodic_noise_removal.get_features(in_image, nw_model)
# calculate the gram matrices for each layer of our style representation
image_grams = {layer: Periodic_noise_removal.find_gram_matrix(image_features[layer]) for layer in image_features}
target = rand_image.clone().requires_grad_(True).to(sys_device)
# for displaying the target image, intermittently
optimizer = optima.Adam([target], lr=params.lr)
steps = 2000
noise_image_extracted = Periodic_noise_removal.train_image(nw_model, optimizer, steps, target, image_grams, params.show_every)
filtered_image = Periodic_noise_removal.im_convert(abs(in_image - noise_image_extracted))
# display content and final, target image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(Periodic_noise_removal.im_convert(in_image))
ax2.imshow(Periodic_noise_removal.im_convert(target))
ax3.imshow(filtered_image)



