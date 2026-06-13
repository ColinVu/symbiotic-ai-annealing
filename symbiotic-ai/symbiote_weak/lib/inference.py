#
# Method to perform inference given an embedding an a chroma collection
#

import numpy as np
from chromadb import Collection


# The threshold (in cosine similarity difference) between two embeddings to consider them the same
DETECT_THRESHOLD: float = 0.05

def perform_inference(
    embedding: np.ndarray,
    collection: Collection
) -> tuple[str, int]:
    """Perform inference on the embedding by querying the collection and deciding the label
    based on the resulting queried vectors. Returns (predicted_item, num_matches)."""
    try:
        n = collection.count()
    except Exception:
        n = 0
    if n == 0:
        return ("unknown", 0)

    emb = embedding if isinstance(embedding, list) else embedding.tolist()
    results = collection.query(query_embeddings=[emb], n_results=min(n, 100))
    ids_0 = results["ids"][0] or []
    meta_0 = results["metadatas"][0] or []
    dist_0 = results["distances"][0] if results.get("distances") else []

    picklists = []
    for i in range(len(ids_0)):
        if i < len(dist_0) and dist_0[i] <= DETECT_THRESHOLD and i < len(meta_0) and meta_0[i]:
            pl = meta_0[i].get("picklist")
            if pl is not None:
                picklists.append(pl)

    bins = {}
    for picklist in picklists:
        for c in str(picklist):
            bins[c] = bins.get(c, 0) + 1
    if not bins:
        return ("unknown", len(picklists))
    return (max(bins, key=bins.get), len(picklists))
