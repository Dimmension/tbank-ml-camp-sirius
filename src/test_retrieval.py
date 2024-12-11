from collections import defaultdict
import numpy as np
from retrieval import RetrievalSystem
import json
import time


retrieval = RetrievalSystem()

with open(r"data\data_full_spoiled.json", "r") as f:
    data = json.load(f)

val_data = data["val"] + data["oos_val"]
val_dict = {entity[0]: entity[1] for entity in val_data}


def calculate_metrics(dataset, k_values):

    metrics = defaultdict(list)
    presence = 0
    overall = 0

    for query, true_label in dataset.items():
        s = time.time()
        retrieved_labels = retrieval.process_query(query, top_r=max(k_values)*2, top_m=max(k_values))
        
        # Calculate metrics for each k
        for k in k_values:
            top_k_results = retrieved_labels[:k]

            # Recall@k: Is true label in top-k results?
            recall_at_k = int(true_label in top_k_results)
            metrics[f"Recall@{k}"].append(recall_at_k)

            # Precision@k: Fraction of relevant results in top-k
            relevant_count = sum(1 for label in top_k_results if label == true_label)
            precision_at_k = relevant_count / k
            metrics[f"Precision@{k}"].append(precision_at_k)

            if true_label in top_k_results:
                presence += 1
            overall += 1
            

        # Mean Average Precision (MAP)
        average_precision = 0
        relevant_count = 0
        for rank, label in enumerate(retrieved_labels, start=1):
            if label == true_label:
                relevant_count += 1
                average_precision += relevant_count / rank
        
        if relevant_count > 0:
            metrics["MAP"].append(average_precision / relevant_count)
        else:
            metrics["MAP"].append(0)

        e = time.time()
        print(f"Time: {e-s}")

    print("Percent of present true labels:", presence / overall)
    # Aggregate metrics by averaging across all queries
    aggregated_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    return aggregated_metrics


# Example Usage

# Define k values for metrics
k_values = [20]

# Initialize the retrieval system
data_path = r"data\labels_with_description.json"
retrieval_system = RetrievalSystem(data_path)

# Calculate metrics
results = calculate_metrics(val_dict, k_values)

# Print results
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
