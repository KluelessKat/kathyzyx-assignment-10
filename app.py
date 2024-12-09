import os
from flask import Flask, render_template, request, send_from_directory
import torch
import open_clip
from open_clip import create_model_and_transforms, tokenizer
from PIL import Image
import pickle
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from preprocess import load_flattened_images, pca_transform


app = Flask(__name__)
app.config["IMAGE_FOLDER"] = "./data/coco_images_resized/"

# Precomputed image embeddings and PCA vectors
with open('./data/image_embeddings_pca.pickle', 'rb') as f:
    df = pd.read_pickle(f)
image_embs = np.stack(df["embedding"])  # Nx512
pca_embs = np.stack(df["pca"])  # Nx50

# CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
pretrained = "openai"
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

# PCA Model
with open("./data/pca.pkl", "rb") as f:
    pca_model = pickle.load(f)


## Helpers

def get_image_embedding(image_path, use_pca=False):
    if use_pca:
        image_flat_arr, _ = load_flattened_images([pathlib.Path(image_path)], verbose=False)
        image_features = pca_transform(pca_model, image_flat_arr)  # 1xK
    else:
        image = preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)  # 1x512
            image_features = image_features.cpu().numpy()
    
    # Normalize to unit vector
    # image_features = image_features.flatten()
    image_features = image_features / np.linalg.norm(image_features)  # shape: 1xF
    return image_features


def get_text_embedding(text):
    text = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features = text_features.cpu().numpy()  # .flatten()
    text_features = text_features / np.linalg.norm(text_features)
    return text_features


def search_images(
    image_embedding=None, 
    text_embedding=None, 
    weight=0.5, 
    use_pca=False, 
    top_k=5
):
    
    print(f"Weight: {weight}")
    
    if image_embedding is not None and text_embedding is not None:
        # Hybrid search
        # image_distances = euclidean_distances(image_embedding, pca_embs if use_pca else image_embs)[0]  # shape: (N,)
        # text_distances = euclidean_distances(text_embedding, image_embs)[0]  # shape: (N,)
        # distances = weight * image_distances + (1 - weight) * text_distances
        
        if use_pca:
            image_distances = cosine_similarity(image_embedding, pca_embs)[0]  # shape: (N,)
            text_distances = cosine_similarity(text_embedding, image_embs)[0]  # shape: (N,)
            similarities = weight * image_distances + (1 - weight) * text_distances
        else:
            joint_embedding = weight * image_embedding + (1 - weight) * text_embedding 
            joint_embedding = joint_embedding / np.linalg.norm(joint_embedding)
            similarities = cosine_similarity(joint_embedding, image_embs)[0]
        
        
    elif image_embedding is not None:
        # Image search
        similarities = cosine_similarity(image_embedding, pca_embs if use_pca else image_embs)[0]
    else:
        # Text search
        similarities = cosine_similarity(text_embedding, image_embs)[0]
    top_indices = np.argsort(similarities)[-top_k:]
    
    # similarities = 1 / (1 + distances)  # There are other ways to convert, but this is common
    # similarities = 1 - distances
    
    results = []
    for idx in top_indices:
        results.append({
            "filename": df.iloc[idx]["file_name"],
            "similarity": similarities[idx],
            # "distance": distances[idx],
        })
    return results


## Flask Endpoints

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(app.config["IMAGE_FOLDER"], filename)


@app.get("/")
def index_get():
    return render_template("index.html", weight=0.5, query_type="hybrid")


@app.post("/")
def index_post():
    os.makedirs("./cache", exist_ok=True)
    
    use_pca = "use_pca" in request.form
    image_embedding = None
    text_embedding = None
    
    query_type = request.form.get("query_type", "hybrid")
    text_query = request.form.get("text")
    image_file = request.files["image"]
    weight = request.form.get("weight", "0.5")
    
    print(f"Query type: {query_type}")
    print(f"Text query: {text_query}")
    print(f"Image file: {image_file}")
    
    ## Check for weight errors
    if not 0 <= float(weight) <= 1:
        return render_template("index.html", 
                               error="Weight must be between 0 and 1",
                               text_query=text_query,
                               query_type=query_type,
                               use_pca=use_pca,
                               weight=weight)
        
    ## Validate that we have the necessary embeddings for the query type
    if query_type == "text" and not text_query:
        return render_template("index.html", 
                               error="Please enter a text query",
                               text_query=text_query,
                               query_type=query_type,
                               use_pca=use_pca,
                               weight=weight)
    
    if query_type == "image" and not image_file:
        return render_template("index.html", 
                               error="Please upload an image",
                               text_query=text_query,
                               query_type=query_type,
                               use_pca=use_pca,
                               weight=weight)
    
    if query_type == "hybrid" and (not image_file or not text_query):
        return render_template("index.html", 
                               error="Please provide both image and text for hybrid search",
                               text_query=text_query,
                               query_type=query_type,
                               use_pca=use_pca,
                               weight=weight)
    
    ## Get embeddings
    if "image" in request.files and (query_type in ["image", "hybrid"]):
        
        if image_file:
            file_path = os.path.join("./cache/temp_image.jpg")
            image_file.save(file_path) 
            image_embedding = get_image_embedding(file_path, use_pca)
            os.remove(file_path)
    
    if request.form.get("text") and (query_type in ["text", "hybrid"]):
        text_embedding = get_text_embedding(text_query)
    
    results = search_images(
        image_embedding=image_embedding,
        text_embedding=text_embedding,
        weight=float(weight),
        use_pca=use_pca
    )
    
    
    return render_template("index.html", 
                           results=results,
                           text_query=text_query,
                           query_type=query_type,
                           use_pca=use_pca,
                           weight=weight)


if __name__ == "__main__":
    app.run(debug=True)


