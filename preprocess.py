
import os
import pathlib
from typing import List
from PIL import Image
import numpy as np 
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
import pandas as pd
import pickle


COCO_IMAGES_PATH = "./data/coco_images_resized/"
EMB_PICKLE_PATH = "./data/image_embeddings.pickle"
TARGET_SIZE = (224, 224)


def get_image_paths(coco_path) -> List[pathlib.Path]:
    coco_path = pathlib.Path(coco_path).absolute()
    assert coco_path.is_dir()

    image_paths = []   
    for image_name in os.listdir(coco_path):
        if image_name.startswith(".") or not image_name.endswith(".jpg"):
            continue
        image_paths.append(coco_path / image_name)
        
    return image_paths
        

def pca_fit(images_arr: np.ndarray, k=50):
    pca = PCA(n_components=k)
    print(f"Training PCA(k={k}) on {images_arr.shape[0]} images.")
    pca.fit(images_arr)
    print("Done.")
    return pca


def pca_transform(pca, images_arr: np.ndarray):
    return pca.transform(images_arr)


def load_flattened_images(image_paths: List[pathlib.Path], verbose=True):
    image_flat_arrs, image_names = [], []
    if verbose:
        print(f"Loading and flattening {len(image_paths)} images..")
    for image_path in tqdm(image_paths, disable=not verbose):
        image_arr = load_image(image_path, 
                               target_size=(224, 224),
                                gray=True)
        image_arr_norm = np.asarray(image_arr, dtype=np.float32) / 255.
        image_flat_arrs.append(image_arr_norm.flatten())
        image_names.append(image_path.name)

    ret_arrs = np.array(image_flat_arrs)
    return ret_arrs, image_names
         

def load_image(image_path, target_size=(224, 224), gray=False) -> np.ndarray:
    img = Image.open(str(image_path))
    if gray:
        img = img.convert("L")
    if target_size is not None:
        img = img.resize(target_size)
    img_arr = np.asarray(img)
    return img_arr




if __name__ == "__main__":
    image_paths = get_image_paths(COCO_IMAGES_PATH)
    # Type: DataFrame to get rid of pylance typing complaints
    image_embeddings_df= pd.read_pickle(EMB_PICKLE_PATH)
    assert len(image_paths) == len(image_embeddings_df)
    assert set([p.name for p in image_paths]) == set(image_embeddings_df["file_name"])
    
    # Train PCA
    fit_pca_image_paths = random.sample(image_paths, k=2000)
    fit_pca_image_arrs, fit_pca_image_names = load_flattened_images(fit_pca_image_paths, verbose=True)
    pca = pca_fit(fit_pca_image_arrs)

    # Get PCA Mapping for Entire Dataset
    image_name_2_pca = {}
    print(f"Transforming {len(image_paths)} images via PCA..")
    for i, image_path in enumerate(tqdm(image_paths)):
        image_flat_arr, _ = load_flattened_images([image_path], verbose=False)
        image_pca = pca_transform(pca, image_flat_arr)  # 1xK
        image_name_2_pca[image_path.name] = image_pca[0] / np.linalg.norm(image_pca[0])
    image_embeddings_df["pca"] = image_embeddings_df["file_name"].apply(
        lambda fn: image_name_2_pca.get(fn, None))
    
    # Save to Pickle (appends column to image_embeddings)
    df_save_path = os.path.join(os.path.dirname(EMB_PICKLE_PATH),
                                "image_embeddings_pca.pickle")
    print(f"Saving new df to: {df_save_path}")
    image_embeddings_df.to_pickle(df_save_path)
    

    # Save model 
    pca_model_save_path = "./data/pca.pkl"
    print(f"Saving PCA model to: {pca_model_save_path}")
    with open(pca_model_save_path, "wb") as f:
        pickle.dump(pca, f)
    
    
    