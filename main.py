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
show_every = 400

# iteration hyperparameters
optimizer = optima.Adam([target], lr=params.lr)
steps = 2000  # decide how many iterations to update your image (5000)

# ## Display the Target Image

# In[31]:


# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))

# In[ ]:




