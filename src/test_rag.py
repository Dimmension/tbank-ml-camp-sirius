# from retrieval_bge_cos_sim import RetrievalSystem_large
from retrieval_bge_faiss_bm25 import RetrievalSystem_faiss_bm25
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import json


def calculate_metrics(dataset):
    tp, tn, fp, fn = 0, 0, 0, 0
    top_k = 10
    print(f"top_k: {top_k}")

    for row in dataset:
        # query = row["query"]
        initial_label = row["initial label"]
        predicted_label = row["predicted label"]

        if initial_label == 'oos':
            if len(predicted_label) == 0:
                tn += 1
            else:
                fp += 1
        else:
            if initial_label == predicted_label:
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


def calculate_metrics_sklearn(dataset):
    true_labels = [row["initial label"] for row in dataset]
    predicted_labels = [row["predicted label"] for row in dataset]
    
    print(classification_report(true_labels, predicted_labels))

    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
    print(f"Precision: {precision_score(true_labels, predicted_labels, average='macro')}")
    print(f"Recall: {recall_score(true_labels, predicted_labels, average='macro')}")
    print(f"F1-score: {f1_score(true_labels, predicted_labels, average='macro')}")
    print("----------------------\n")

# Example Usage

if __name__ == '__main__':
    with open(r"data\val_result_full_without_neighbors.json", "r") as f:
        data = json.load(f)

    # for item in data:
    #     print(item["query"])
    #     print(item["initial label"])
    #     print(item["predicted label"])
    #     break

    calculate_metrics(data)
    calculate_metrics_sklearn(data)
