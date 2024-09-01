import os
import random
import matplotlib.pyplot as plt # type: ignore
from PIL import Image
import torchvision.transforms as transforms

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