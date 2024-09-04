import os
import random
import matplotlib.pyplot as plt # type: ignore
from PIL import Image
import torchvision.transforms as transforms
import torch
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def save_images(dataset, split_name,output_dir,preprocess=None):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    for idx, sample in enumerate(dataset):
        image=sample["image"]
        if preprocess is not None:
            image = preprocess(image)
        # Convert back to PIL image for saving
            image = transforms.ToPILImage()(image)
        image_path = os.path.join(split_dir, f'image_{idx}.png')
        image.save(image_path,"jpeg")
# Save preprocessed training images
def plot_original_and_reconstructed(model, dataloader, device, num_images=5):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    images = []
    reconstructed_images = []
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data = data.to(device)  # Move data to the same device as the model
            
            # Adjust based on the model architecture
            output = model(data)
            
            if len(output) == 2:
                    # This is likely a standard autoencoder (encoded, reconstructed)
                    reconstructed = output[1]
            elif len(output) == 3:
                    # This is likely a VAE (reconstructed, mu, logvar)
                    reconstructed = output[0]

            # Move tensors back to CPU for visualization
            data = data.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()

            # Debug: print shapes
            #print(f"Original shape: {data.shape}, Reconstructed shape: {reconstructed.shape}")

            images.extend(data[:num_images])  # Original images
            reconstructed_images.extend(reconstructed[:num_images])  # Reconstructed images
            
            if len(images) >= num_images:
                break

    # Plot the original and reconstructed images side by side
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        original_image = images[i].reshape(3, 50, 50).transpose(1, 2, 0)
        plt.imshow(original_image)
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed images
        plt.subplot(2, num_images, num_images + i + 1)
        print(f"Reconstructed Image {i} shape: {reconstructed_images[i].shape}")  # Print the shape for debugging
        # Adjust reshape based on actual reconstructed size
        try:
            if reconstructed_images[i].size == 3 * 50 * 50:
                reconstructed_image = reconstructed_images[i].reshape(3, 50, 50).transpose(1, 2, 0)
            else:
                # Handle other cases; in VAEs this might differ based on the latent space size and output
                reconstructed_image = reconstructed_images[i].reshape(-1, 50, 50).transpose(1, 2, 0)
        except ValueError as e:
            print(f"Error reshaping image {i}: {e}")
            continue
        
        plt.imshow(reconstructed_image)
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()

# Assuming you want to plot 5 images



def plot_random_images(dataset_path, num_images=15, rows=3, cols=5):

    images=[]
    for element in os.listdir(dataset_path):
        images.append(os.path.join(dataset_path, element))
    # Get random indices
    random_indices = random.sample(range(len(images)), num_images)
    plt.figure(figsize=(15, 9))
    for i, idx in enumerate(random_indices):
        #image = dataset[idx]['image']
        img = Image.open(images[i])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
def vae_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
   # print(f'Reconstruction Loss: {reproduction_loss.item()}, KL Divergence: {KLD.item()}')

    return reproduction_loss + KLD
def generate_new_images_vae(model, num_images, latent_dim, device):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        # Sample from the standard normal distribution
        z = torch.randn(num_images, latent_dim).to(device)
        
        # Pass through the decoder part of the VAE
        generated_images = model.decode(z)  # Ensure your VAE has a decode method

        # Move tensors back to CPU for visualization
        generated_images = generated_images.cpu().numpy()

    # Plot the generated images
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i].transpose(1, 2, 0))  # Transpose for RGB display
        plt.title("Generated")
        plt.axis('off')
    
    plt.show()

# Assuming your VAE has a latent dimension of 512

def train_autoencoder(model, train_data, val_data, device, epochs=100, lr=1e-4, ckpt_dir="checkpoints", arch_name="model", loss_fn="mse"):
    """
    Train an autoencoder or Variational Autoencoder (VAE) using PyTorch.

    Parameters:
    - model: The autoencoder or VAE model to be trained.
    - train_data: The training dataset.
    - val_data: The validation dataset.
    - device: The device to run the training on (CPU or GPU).
    - epochs (int): The number of training epochs. Default is 100.
    - lr (float): The learning rate for the optimizer. Default is 1e-4.
    - ckpt_dir (str): The directory to save checkpoints. Default is "checkpoints".
    - arch_name (str): The name of the architecture. Default is "model".
    - loss_fn (str): The loss function to use. Can be either "mse" for Mean Squared Error or "vae" for Variational Autoencoder loss. Default is "mse".

    Returns:
    None
    """
    # Create a directory for saving checkpoints with the architecture name
    crt_time = time.strftime("%Y%m%d_%H%M%S")
    dir_ckpt = os.path.join(ckpt_dir, f"{crt_time}_{arch_name}")
    os.makedirs(dir_ckpt, exist_ok=True)

    # Set up TensorBoard writer
    writer = SummaryWriter()

    # Move model to the appropriate device
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Select the appropriate loss function
    if loss_fn == "mse":
        loss_function = nn.MSELoss()
    elif loss_fn == "vae":
        loss_function = vae_loss_function
    else:
        raise ValueError("Unsupported loss function type")

    # Training loop
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()  # Set the model to training mode

        with tqdm(train_data, unit="batch", total=len(train_data), desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for batch_idx, data in enumerate(tepoch):
                data = data.to(device)

                # Forward pass
                if loss_fn == "vae":
                    x_hat, mean, logvar = model(data)
                    loss = loss_function(data, x_hat, mean, logvar)
                else:
                    encoded, decoded = model(data)
                    loss = loss_function(decoded, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * data.size(0)
                tepoch.set_postfix(train_loss=total_train_loss / ((batch_idx + 1) * data.size(0)))
                writer.add_scalar("Loss/train", loss.item(), epoch)

        epoch_train_loss = total_train_loss / len(train_data.dataset)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            for data in val_data:
                data = data.to(device)

                if loss_fn == "vae":
                    x_hat, mean, logvar = model(data)
                    val_loss = loss_function(data, x_hat, mean, logvar)
                else:
                    encoded, decoded = model(data)
                    val_loss = loss_function(decoded, data)

                total_val_loss += val_loss.item() * data.size(0)

        epoch_val_loss = total_val_loss / len(val_data.dataset)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)

        print(f"Epoch {epoch+1}/{epochs} : train_loss = {epoch_train_loss:.4f}, val_loss = {epoch_val_loss:.4f}")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            ckpt_file = os.path.join(dir_ckpt, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, ckpt_file)

    writer.close()