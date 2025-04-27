import open_clip
from PIL import Image
import torch
import os
import numpy as np
from pathlib import Path

# path
image_folder = '/home/happyhome/Documents/Quocanh/Fashion-Semantic-Search/images'
model_path = '/home/happyhome/Documents/Quocanh/Fashion-Semantic-Search/pretrained/finetuned_clip.pt'

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load(model_path, map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)

image_paths = list(Path(image_folder).glob('*.jpg'))

# Output paths for storing embeddings and image paths
embeddings_file = 'vector_data/image_embeddings.npy'
paths_file = 'vector_data/image_paths.txt'

# Function to generate image embeddings
def generate_image_embeddings(image_paths, batch_size=32):
    embeddings = []
    valid_image_paths = []
    
    for i in range(0, len(image_paths), batch_size):
        print(i)
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []
        
        # Load and preprocess images
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).to(device)
                batch_images.append(img)
                batch_valid_paths.append(str(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into a batch
        batch_images = torch.stack(batch_images)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = clip_model.encode_image(batch_images)
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)  # Normalize
            embeddings.append(batch_embeddings.cpu().numpy())
            valid_image_paths.extend(batch_valid_paths)
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, valid_image_paths

# Generate and save embeddings
print("Generating image embeddings...")
image_embeddings, valid_image_paths = generate_image_embeddings(image_paths)
print(f"Processed {len(valid_image_paths)} images.")

# Save embeddings and image paths
np.save(embeddings_file, image_embeddings)
with open(paths_file, 'w') as f:
    f.write('\n'.join(valid_image_paths))
print(f"Saved embeddings to {embeddings_file}")
print(f"Saved image paths to {paths_file}")