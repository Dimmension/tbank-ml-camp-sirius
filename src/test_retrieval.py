from retrieval_bge import RetrievalSystem_large
# from retrieval_minilm import RetrievalSystem_small
from retrieval_minilm_cos_sim import RetrievalSystem_small
import json
import time
import tqdm


def calculate_metrics(dataset):
    presence = 0
    overall = 0

    top_k = 10
    print(f"top_k: {top_k}")
          
    for query, true_label in tqdm.tqdm(dataset.items()):
        # retrieved_labels = retrieval.process_query(query, top_r=top_r, top_m=top_m)
        retrieved_labels = retrieval.process_query(query, top_k=top_k)
        
        if true_label == 'oos':
            if len(retrieved_labels) == 0:
                presence += 1
        if true_label in retrieved_labels:
            presence += 1
        overall += 1

    print(f"\nPercent of present true labels: {presence / overall:.4f}")
    return presence / overall


# Example Usage

if __name__ == '__main__':    
    # retrieval = RetrievalSystem_large()
    retrieval = RetrievalSystem_small()

    with open(r"data\data_full_spoiled.json", "r") as f:
        data = json.load(f)

    val_data = data["val"] #+ data["oos_val"]
    val_dict = {entity[0]: entity[1] for entity in val_data}
    calculate_metrics(val_dict)
