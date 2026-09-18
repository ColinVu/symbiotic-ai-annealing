import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer
from scipy.optimize import linear_sum_assignment

HF_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_XLSX_PATH = "randomized_Object_List.xlsx"
DESC_COLUMN = "Text Description"


def load_label_descriptions(
    xlsx_path: str = DEFAULT_XLSX_PATH,
    desc_column: str = DESC_COLUMN,
) -> Dict[str, str]:
    """shelf="C", row=4, column=1 -> key "c41" mapped to its text description."""
    import pandas as pd

    df = pd.read_excel(xlsx_path)
    cols = {str(c).strip(): c for c in df.columns}

    def _num_str(value) -> str:
        f = float(value)
        return str(int(f)) if f.is_integer() else str(value).strip()

    descriptions: Dict[str, str] = {}
    for _, row in df.iterrows():
        shelf, r, c = row[cols["Shelf"]], row[cols["Row"]], row[cols["Column"]]
        if pd.isna(shelf) or pd.isna(r) or pd.isna(c):
            continue
        name = f"{str(shelf).strip().lower()}{_num_str(r)}{_num_str(c)}"
        desc = row[cols[desc_column]]
        descriptions[name] = str(desc).strip() if not pd.isna(desc) else name

    return descriptions


def encode_label_texts(
    label_texts: Sequence[str],
    label_descriptions: Dict[str, str],
    model_name: str = HF_MODEL_NAME,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [label_descriptions.get(lbl, lbl) for lbl in label_texts]
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True) # normalize

    feats = feats.cpu().numpy().astype(np.float64)
    return {lbl: feats[i] for i, lbl in enumerate(label_texts)}


def _segment_cost_row(
    seg,
    T_matrix: np.ndarray,
    aggregation: str = "max",
) -> np.ndarray:
    em = np.atleast_2d(np.asarray(seg.embeddings, dtype=np.float64))
    norms = np.linalg.norm(em, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    em = em / norms

    # Shape: (number of embeddings/frames, number of candidate labels).
    # Each column contains every frame's comparison to one candidate.
    similarities = em @ T_matrix.T
    distances = 1.0 - similarities

    if aggregation == "sum":
        return np.sum(distances, axis=0)
    if aggregation == "mean":
        return np.mean(distances, axis=0)
    if aggregation == "max":
        # Maximum cosine similarity == minimum cosine distance, so each
        # candidate uses its own closest frame from the segment.
        return 1.0 - np.max(similarities, axis=0)

    raise ValueError(
        f"Unknown aggregation mode {aggregation!r}; expected 'sum', 'mean', or 'max'."
    )


def clip_initialization(
    all_segments: List,
    text_embeddings: Optional[Dict[str, np.ndarray]] = None,
    model_name: str = HF_MODEL_NAME,
    device: Optional[str] = None,
    xlsx_path: str = DEFAULT_XLSX_PATH,
    aggregation: str = "max",
    video_picklists: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Per-video CLIP init + Hungarian assignment.

    When ``video_picklists`` is supplied, its explicit flat JSON picklist is
    used as the candidate-slot multiset. This preserves duplicate SKUs and
    repeated identical subsets. Without it, the historical candidate-pool
    construction is retained for backward compatibility.

    ``aggregation`` controls how frame-level comparisons become one cost per
    segment/candidate pair:
      - ``"sum"``: sum all frame-to-candidate cosine distances (original logic)
      - ``"mean"``: average all frame-to-candidate cosine distances
      - ``"max"``: use maximum cosine similarity (the closest frame)

    Raises ValueError if a video's segment count doesn't match its candidate
    slot count or if an explicit ``video_picklists`` mapping is missing an
    entry.
    """
    if text_embeddings is None:
        label_descriptions = load_label_descriptions(xlsx_path)
        text_embeddings = encode_label_texts(
            list(label_descriptions.keys()), label_descriptions,
            model_name=model_name, device=device,
        )

    groups = defaultdict(list)
    for s in all_segments:
        groups[s.video_id].append(s)

    labels: Dict = {}
    for vid_id, segs in groups.items():
        segs = sorted(segs, key=lambda x: x.segment_id)

        if video_picklists is not None:
            if vid_id not in video_picklists:
                raise ValueError(f"video={vid_id}: missing flat picklist in video_picklists")
            cand = [str(x) for x in video_picklists[vid_id]]
        else:
            # Historical behavior: merge the candidate multisets associated
            # with this video's segments into one global candidate pool.
            unique_multisets = {s.candidate_labels for s in segs}
            if len(unique_multisets) == 1:
                cand = list(next(iter(unique_multisets)))
            else:
                merged = Counter()
                for ms in unique_multisets:
                    merged += Counter(ms)
                cand = list(merged.elements())

        if len(segs) != len(cand):
            raise ValueError(
                f"video={vid_id}: {len(segs)} segments but {len(cand)} "
                f"candidate labels — Hungarian assignment requires N == M."
            )

        T_matrix = np.stack([text_embeddings[c] for c in cand])
        cost = np.stack(
            [_segment_cost_row(s, T_matrix, aggregation=aggregation) for s in segs]
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            labels[segs[i].label_key] = cand[j]

    return labels
