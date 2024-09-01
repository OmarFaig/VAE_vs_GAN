import os
import random
import matplotlib.pyplot as plt # type: ignore
from PIL import Image
import torchvision.transforms as transforms
import torch
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
def plot_original_and_reconstructed(model, dataloader, device,num_images=5):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    images = []
    reconstructed_images = []
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            data = data.to(device)  # Move data to the same device as the model
            _, reconstructed = model(data)
            
            # Move tensors back to CPU for visualization
            data = data.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()
            print(f"Original shape: {data.shape}, Reconstructed shape: {reconstructed.shape}")  # Debugging output

            images.extend(data[:num_images])  # Original images
            reconstructed_images.extend(reconstructed[:num_images])  # Reconstructed images
            
            if len(images) >= num_images:
                break

    # Plot the original and reconstructed images side by side
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].reshape(3, 50, 50).transpose(1, 2, 0))  # Reshape and transpose for RGB
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed images
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed_images[i].reshape(3, 50, 50).transpose(1, 2, 0))  # Reshape and transpose for RGB
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.show()
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