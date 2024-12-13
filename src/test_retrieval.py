from retrieval_bge_cos_sim import RetrievalSystem_chroma
from retrieval_bge_faiss_bm25 import RetrievalSystem_faiss_bm25
import json
import time
import tqdm
import itertools


def calculate_metrics(dataset,
                      embedder_name="chinchilla04/bge-finetuned-train",
                      retriever="faiss_bm25",
                      threshold_mean_diff=1e-6,
                      threshold_conf_drop=1e-5):
    print("Retriever:", retriever)
    print("Embedder:", embedder_name)
    if retriever == "cos_sim":
        retrieval = RetrievalSystem_chroma(embedder_name=embedder_name,
                                    threshold_mean_diff=threshold_mean_diff,
                                    )
        top_r = 10
        top_k = 10
        
    elif retriever == "faiss_bm25":
        retrieval = RetrievalSystem_faiss_bm25(embedder_name=embedder_name,
                                    threshold_mean_diff=threshold_mean_diff,
                                    )
        top_r = 20
        top_k = 10

    # retrieval = RetrievalSystem_chroma(threshold_mean_diff=threshold_mean_diff,
    #                             #   threshold_conf_drop=threshold_conf_drop,
    #                             )
    print(f"Threshold mean diff: {threshold_mean_diff}")
    # print(f"Threshold conf drop: {threshold_conf_drop}")
    
    tp, tn, fp, fn = 0, 0, 0, 0
    print(f"top_r: {top_r}")
    print(f"top_k: {top_k}")

    for query, true_label in tqdm.tqdm(dataset.items()):
        retrieved_labels = retrieval.process_query(query, top_r=top_r,top_k=top_k)

        if true_label == 'oos':
            if len(retrieved_labels) == 0:
                tn += 1
            else:
                fp += 1
        else:
            # if true_label in retrieved_labels:
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
    embedders = ["chinchilla04/bge-finetuned-train", "BAAI/bge-large-en-v1.5"]
    retriever = ["cos_sim", "faiss_bm25"]
    # retriever = ["faiss_bm25"]
    for emb, retr in itertools.product(embedders, retriever):
        calculate_metrics(dataset, embedder_name=emb, retriever=retr)


# Example Usage

if __name__ == '__main__':    
    with open(r"data\data_full_spoiled.json", "r") as f:
        data = json.load(f)

    val_data = data["val"] + data["oos_val"]
    val_dict = {entity[0]: entity[1] for entity in val_data}

    # calculate_metrics(val_dict)

    hp_tuning(val_dict)
