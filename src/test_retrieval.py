# from retrieval_bge_cos_sim import RetrievalSystem_large
from retrieval_bge_faiss_bm25 import RetrievalSystem_large
import json
import time
import tqdm
import itertools


def calculate_metrics(dataset, threshold_mean_diff=1e-6, threshold_conf_drop=1e-5):
    retrieval = RetrievalSystem_large(threshold_mean_diff=threshold_mean_diff,
                                    #   threshold_conf_drop=threshold_conf_drop,
                                      )
    print(f"Threshold mean diff: {threshold_mean_diff}")
    # print(f"Threshold conf drop: {threshold_conf_drop}")
    
    tp, tn, fp, fn = 0, 0, 0, 0
    top_k = 10
    print(f"top_k: {top_k}")

    for query, true_label in tqdm.tqdm(dataset.items()):
        retrieved_labels = retrieval.process_query(query, top_k=top_k)

        if true_label == 'oos':
            if len(retrieved_labels) == 0:
                tn += 1
            else:
                fp += 1
        else:
            if len(retrieved_labels) > 0:
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

def hp_tuning(dataset):
    t_mean_diff = [1e-6, 5e-7, 1e-7, 5e-8]
    t_conf_drop = [1e-5, 5e-6, 1e-6, 5e-7]

    for t1, t2 in itertools.product(t_mean_diff, t_conf_drop):
        calculate_metrics(val_dict, t1, t2)


# Example Usage

if __name__ == '__main__':    
    with open(r"data\data_full_spoiled.json", "r") as f:
        data = json.load(f)

    val_data = data["val"] + data["oos_val"]
    # val_data = data["val"]
    # val_data = data["oos_val"]
    # val_data = data["val"][:20] + data["oos_val"][:20]
    
    val_dict = {entity[0]: entity[1] for entity in val_data}

    calculate_metrics(val_dict)
