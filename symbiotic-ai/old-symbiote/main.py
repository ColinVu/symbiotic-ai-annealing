#
# The symbiote is an ego-centric AI that will learn to order pick by observation
# 

from transformers import AutoModel, AutoProcessor
import chromadb
import cv2
import argparse
import sys
import numpy as np

from lib.embedding import MODEL, embed_image, add_embedding_to_collection
from lib.inference import perform_inference
from lib.state_detection import detect_state, State


def main(
    model: AutoModel,
    processor: AutoProcessor,
    collection: chromadb.Collection,
    capture: cv2.VideoCapture,
    picklist: str,
    outfile: str
):
    f = open(outfile, "a")
    f.write(f"Picklist: {picklist}")
    id = collection.count()
    embeddings = []
    try:
        while True:
            img = capture.read()
            img = cv2.resize(img, (1920, 1080))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            current_state = detect_state(img)
            match current_state:
                case State.PICK:
                    pass
                case State.CARRY_WITH:
                    embeddings.append(embed_image(img, model, processor))
                case State.PLACE:
                    if len(embeddings) > 0:
                        average_embedding = np.average(np.array(embeddings))
                        (item_prediction, _) = perform_inference(average_embedding, collection)
                        f.write(item_prediction)
                        add_embedding_to_collection(average_embedding, id, picklist, collection)
                        embeddings = []
                        id += 1
                case State.CARRY_WITHOUT:
                    pass
    except KeyboardInterrupt:
        f.write("---")
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, help="Path to the input video", required=False, default="")
    parser.add_argument("-c", "--capture", type=str, help="Interface to start the video capture on", required=False, default="")
    parser.add_argument("-p", "--picklist", type=str, help="The picklist being picked", required=True)
    parser.add_argument("-o", "--outfile", type=str, help="The file to write predictions to", required=True)
    args = parser.parse_args()
    if args.video == "" and args.capture == "":
        print("Invalid Arguments: Please provide a video file to read or an input stream")
        sys.exit(1)
    capture: cv2.VideoCapture
    if args.video != "":
        capture = cv2.VideoCapture(args.video)
    elif args.capture != "":
        capture = cv2.VideoCapture(args.capture)
    model = AutoModel.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL)
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="symbiotic-ai",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
        metadata={
            "description": "vector database for the AI-through-symbiosis project"
        }
    )
    main(model, processor, collection, capture, args.picklist, args.outfile)
