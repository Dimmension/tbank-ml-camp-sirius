# from retrieval_bge_cos_sim import RetrievalSystem_large
from retrieval_bge_faiss_bm25 import RetrievalSystem_large
import json
import time
import tqdm
import itertools


def calculate_metrics(dataset: dict, dataset_new: dict):
    tp, tn, fp, fn = 0, 0, 0, 0
    top_k = 10
    print(f"top_k: {top_k}")

    for query, true_label in tqdm.tqdm(dataset.items()):
        retrieved_labels = dataset_new[query]

        if true_label == 'oos':
            if len(retrieved_labels) == 0:
                tn += 1
            else:
                fp += 1
        else:
            if true_label in retrieved_labels:
            # if len(retrieved_labels) > 0:
                tp += 1
            else:
                fn += 1

    print(f"\nAccurcy: {(tn + tp) / (tn + tp + fp + fn):.4f}")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.4f}")
    else:
        print("Recall: Undefined (no positive samples)")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.4f}")
    else:
        print("Precision: Undefined (no predicted positives)")

    print(f"F1-score: {2*recall*precision / (recall + precision):.4f}")

    print(f"TP = {tp}")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print("----------------------\n")


# Example Usage

if __name__ == '__main__':    
    with open(r"data\data_full_spoiled.json", "r") as f:
        data = json.load(f)

    val_data = data["val"] + data["oos_val"]
    val_dict = {entity[0]: entity[1] for entity in val_data}

    # прогон по ллмке
    # val_dict.keys это запросы, values это лэйблы
    # val_dict_new values это лэйблы, предсказанные ллкмкой
    val_dict_new = {}

    calculate_metrics(val_dict, val_dict_new)
