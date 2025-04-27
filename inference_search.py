import open_clip
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the CLIP model
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('pretrained/finetuned_clip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Paths to stored embeddings and image paths
embeddings_file = 'vector_data/image_embeddings.npy'
paths_file = 'vector_data/image_paths.txt'

# Load embeddings and image paths
image_embeddings = np.load(embeddings_file)
with open(paths_file, 'r') as f:
    image_paths = f.read().splitlines()

print(f"Loaded {len(image_paths)} image embeddings.")

# Function to perform vector search
def search_images(query, top_k=5):
    # Tokenize and encode the text query
    prompt = "a photo of a"
    text_input = [prompt + " " + query]
    tokenized_text = tokenizer(text_input).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
    
    # Compute cosine similarity
    text_features = text_features.cpu().numpy()
    similarities = np.dot(image_embeddings, text_features.T).squeeze()
    
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    top_k_scores = similarities[top_k_indices]
    top_k_paths = [image_paths[i] for i in top_k_indices]
    
    return list(zip(top_k_paths, top_k_scores))


# Function to visualize and save results
def visualize_results(query, results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot
    n_results = len(results)
    cols = min(n_results, 3)  # Up to 3 columns
    rows = (n_results + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"Search Results for Query: '{query}'", fontsize=16)
    
    # Flatten axes for easy iteration
    axes = np.array(axes).flatten() if n_results > 1 else [axes]
    
    # Plot each image
    for i, (img_path, score) in enumerate(results):
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].set_title(f"{Path(img_path).name}\nSimilarity: {score:.4f}", fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            axes[i].set_title("Error loading image")
            axes[i].axis('off')
    
    # Turn off unused axes
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    output_file = os.path.join(output_dir, f"results_{query.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.show()
    plt.close()

# Main function for command-line usage
def main():
    parser = argparse.ArgumentParser(description="Search images using a text query and visualize results.")
    parser.add_argument('--query', type=str, required=True, help="Text query to search images")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top results to return")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save visualization")
    args = parser.parse_args()
    
    results = search_images(args.query, args.top_k)
    
    print(f"\nTop {args.top_k} results for query: '{args.query}'")
    for i, (img_path, score) in enumerate(results, 1):
        print(f"{i}. {img_path} (Similarity: {score:.4f})")
    
    # Visualize and save results
    visualize_results(args.query, results, args.output_dir)

if __name__ == "__main__":
    main()