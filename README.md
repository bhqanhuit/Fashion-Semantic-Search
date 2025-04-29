# Fashion Semantic Search

Fashion Semantic Search is a web application that allows users to search for fashion images using text queries. It leverages a fine-tuned CLIP model to perform semantic search on a pre-indexed dataset of fashion images, returning the top-k most similar images along with a visualization plot. We reused the pretrained model from [OpenFashionCLIP](https://github.com/aimagelab/open-fashion-clip)

## Features
- **Text-Based Search**: Input a query to find matching fashion images.
- **Top-K Results**: Specify the number of results to display (1-10).
- **Visualization**: View a plot of the top-k images with their similarity scores.
- **Local Hosting**: Run the app locally with a FastAPI backend and a simple HTML frontend.

## Project Structure
```
fashion-search-app/
├── app.py                    # FastAPI backend
├── index.html                # Frontend HTML page
├── pretrained/
│   └── finetuned_clip.pt     # Fine-tuned CLIP model weights
├── vector_data/
│   ├── image_embeddings.npy  # Precomputed image embeddings
│   └── image_paths.txt       # Relative paths to images (e.g., /images/56983.jpg)
├── results/                  # Directory for generated visualization plots
│   └── (e.g., results_blue_cowl_neck_maxi-dress.png)
├── images/                   # Directory containing image files
│   ├── 56983.jpg
│   ├── blue_maxi_dress_001.jpg
│   └── (other images)
```

## Prerequisites
- **Python 3.8+**
- **Dependencies**: Install the required Python packages:
  ```bash
  pip install fastapi uvicorn open-clip-torch pillow torch numpy matplotlib pydantic
  ```

## Setup Instructions

1. **Verify Data Files**:
   - Ensure `pretrained/finetuned_clip.pt`, `vector_data/image_embeddings.npy`, and `vector_data/image_paths.txt` exist.
   - Confirm that `image_paths.txt` contains relative paths (e.g., `/images/56983.jpg`). If it contains absolute paths, run the `update_image_paths.py` script:
     ```bash
     cd fashion-search-app
     python update_image_paths.py
     ```
   - Verify that the `images/` directory contains the images listed in `image_paths.txt` (e.g., `images/56983.jpg`).

2. **Run the FastAPI Backend**:
   Start the backend server:
   ```bash
   cd fashion-search-app
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   - The server will run at `http://localhost:8000`.
   - Expected output:
     ```
     Loaded 1000 image embeddings.
     INFO:     Started server process [12345]
     INFO:     Application startup complete.
     INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
     ```

3. **Serve the Frontend**:
   Serve the HTML frontend using Python’s built-in HTTP server:
   ```bash
   cd fashion-search-app
   python -m http.server 3000
   ```
   - The frontend will be accessible at `http://localhost:3000`.

## Usage
1. **Access the App**:
   Open a browser and navigate to `http://localhost:3000`.

2. **Perform a Search**:
   - Enter a query in the "Search Query" field (e.g., "blue cowl neck maxi-dress").
   - Set the "Number of Results" (1-10, default is 5).
   - Click the "Search" button.
   - The app will display:
     - A grid of the top-k images with their similarity scores and filenames.
     - A visualization plot below the grid.

3. **API Access**:
   Test the backend directly using `curl`:
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "blue cowl neck maxi-dress", "top_k": 5}'
   ```
   - Example response:
     ```json
     {
       "results": [
         {"path": "/images/56983.jpg", "score": 0.9234},
         {"path": "/images/blue_maxi_dress_001.jpg", "score": 0.9156},
         {"path": "/images/navy_dress_023.jpg", "score": 0.9102},
         {"path": "/images/cowl_dress_045.jpg", "score": 0.9054},
         {"path": "/images/floral_dress_067.jpg", "score": 0.9023}
       ],
       "plot": "results/results_blue_cowl_neck_maxi-dress.png"
     }
     ```
   - View the plot by opening `http://localhost:8000/plot/blue_cowl_neck_maxi-dress` in a browser.