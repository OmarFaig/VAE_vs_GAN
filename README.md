# Image Generation with Variational Autoencoder (VAE)

This project implements three types of autoencoders for image generation:

1. **Linear Autoencoder**: A conventional autoencoder with only linear layers and ReLU activation functions.
2. **Convolutional Autoencoder**: This version uses convolutional layers in place of linear layers.
3. **Variational Autoencoder (VAE)**: A more advanced version that models a latent distribution, allowing for generating new, unseen images.

### 1. **Linear Autoencoder**

The linear autoencoder uses fully connected layers and ReLU activations to compress and reconstruct the input images. It serves as a baseline for comparison with more complex architectures.

### 2. **Convolutional Autoencoder**

In the convolutional autoencoder, convolutional layers are used instead of fully connected layers, allowing the model to better capture spatial hierarchies in the input images.
The results looks like the following:
![alt text](res_imgs/image_conv.png)

### 3. **Variational Autoencoder (VAE)**

The VAE is a probabilistic model that maps the input images to a latent distribution. The encoder outputs mean and variance parameters, which are used to sample from a latent space. The decoder then reconstructs the images based on the sampled latent variables. This architecture enables the generation of new images that are similar to the input data but not identical.
The results looks like the following:
![alt text](res_imgs/image_vae.png)

### 4 . ** Generative Adversarial Networks (GAN)**

Another famous generative model is GAN. Here a DCGAN model is implemented, for implementation some parts of codes from this [repo](https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L18) is used. The results from GAN looks like the following :
![alt text](res_imgs/GAN.png)
