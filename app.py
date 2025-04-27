import open_clip
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mount the images directory to serve images
app.mount("/images", StaticFiles(directory="images"), name="images")

# Load CLIP model and embeddings at startup
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
state_dict = torch.load('pretrained/finetuned_clip.pt', map_location=device)
clip_model.load_state_dict(state_dict['CLIP'])
clip_model = clip_model.eval().requires_grad_(False).to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

embeddings_file = 'vector_data/image_embeddings.npy'
paths_file = 'vector_data/image_paths.txt'

image_embeddings = np.load(embeddings_file)
with open(paths_file, 'r') as f:
    image_paths = f.read().splitlines()

print(f"Loaded {len(image_paths)} image embeddings.")

# Pydantic model for request body
class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

# Function to perform vector search
def search_images(query, top_k=5):
    prompt = "a photo of a"
    text_input = [prompt + " " + query]
    tokenized_text = tokenizer(text_input).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    text_features = text_features.cpu().numpy()
    similarities = np.dot(image_embeddings, text_features.T).squeeze()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    top_k_scores = similarities[top_k_indices]
    top_k_paths = [image_paths[i] for i in top_k_indices]
    
    return list(zip(top_k_paths, top_k_scores))

# Function to visualize and save results
def visualize_results(query, results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    n_results = len(results)
    cols = min(n_results, 3)
    rows = (n_results + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"Search Results for Query: '{query}'", fontsize=16)
    axes = np.array(axes).flatten() if n_results > 1 else [axes]
    
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
    
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_dir, f"results_{query.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

# FastAPI endpoint for search
@app.post("/search")
def search(search_query: SearchQuery):
    try:
        results = search_images(search_query.query, search_query.top_k)
        output_file = visualize_results(search_query.query, results)
        response = {
            "results": [{"path": path, "score": float(score)} for path, score in results],
            "plot": output_file
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to serve plot image
@app.get("/plot/{query}")
def get_plot(query: str):
    plot_path = os.path.join('results', f"results_{query.replace(' ', '_')}.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)