#
# Image embedding helper functions
#

from transformers import AutoModel, AutoProcessor
from chromadb import Collection
import torch
import cv2
import numpy as np

from hand_detection import segment_hand


# The image embedding model to use
MODEL: str = "openai/clip-vit-base-patch32"

def embed_image(
    image: cv2.typing.MatLike,
    model: AutoModel,
    processor: AutoProcessor,
):
    """Embed an image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = segment_hand(image)
    if image is None:
        print(f"Unable to segment image in image")
        return None
    inputs = processor(images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.numpy()[0]

def add_embedding_to_collection(
    embedding: np.ndarray,
    id: int,
    picklist: str,
    collection: Collection,
):
    """Add a computed embedding into the vector database collection"""
    collection.add(
        ids=[f"id{id}"],
        embeddings=[embedding],
        metadatas=[{"picklist": picklist}]
    )
