import numpy as np


def top_k_accuracy(y_true, y_pred_top_k):
    """
    Calculates the top-k accuracy.

    :param y_true: List of ground truth labels. (n_samples,)
    :param y_pred_top_k: List of lists with top k predicted labels for each sample. (n_samples, k)
    :return: Top-k accuracy score.
    """
    if len(y_true) != len(y_pred_top_k):
        raise ValueError("y_true and y_pred_top_k must have the same length.")

    correct_predictions = 0
    for true_label, top_k_preds in zip(y_true, y_pred_top_k):
        if true_label in top_k_preds:
            correct_predictions += 1

    return correct_predictions / len(y_true)


def get_top_k_predictions(search_results, image_labels, k):
    """
    Helper function to extract labels from search results.

    :param search_results: A list of lists, where each inner list contains
                           the identifiers (e.g., filenames) of retrieved images.
    :param image_labels: A dictionary mapping image identifiers to their labels.
    :param k: The number of top results to consider.
    :return: A list of lists with the top k predicted labels for each query.
    """
    y_pred_top_k = []
    for result_set in search_results:
        top_k_labels = [
            image_labels.get(img_id, "unknown") for img_id in result_set[:k]
        ]
        y_pred_top_k.append(top_k_labels)
    return y_pred_top_k


if __name__ == "__main__":
    # This is an example of how you might use the top_k_accuracy function.
    # In a real scenario, y_true would be the ground truth labels of your
    # query images, and y_pred_top_k would come from your image search model.

    # 1. Ground Truth Labels for 5 query images
    y_true_example = ["cat", "dog", "bird", "cat", "fish"]

    # 2. Let's assume your image search returns a list of ranked image filenames for each query.
    # These would be the outputs from your search system (e.g., src.core.search)
    search_results_example = [
        [
            "cat_01.jpg",
            "dog_01.jpg",
            "cat_02.jpg",
            "bird_01.jpg",
            "fish_01.jpg",
        ],  # Query 1 results
        [
            "cat_03.jpg",
            "dog_02.jpg",
            "dog_03.jpg",
            "fish_02.jpg",
            "bird_02.jpg",
        ],  # Query 2 results
        [
            "bird_03.jpg",
            "fish_03.jpg",
            "cat_04.jpg",
            "dog_04.jpg",
            "bird_04.jpg",
        ],  # Query 3 results
        [
            "dog_05.jpg",
            "fish_04.jpg",
            "cat_05.jpg",
            "bird_05.jpg",
            "cat_06.jpg",
        ],  # Query 4 results
        [
            "cat_07.jpg",
            "dog_06.jpg",
            "bird_06.jpg",
            "fish_05.jpg",
            "fish_06.jpg",
        ],  # Query 5 results
    ]

    # 3. You need a mapping from your image filenames/IDs to their actual labels.
    # This would typically be loaded from your dataset information.
    image_labels_mapping = {
        "cat_01.jpg": "cat",
        "cat_02.jpg": "cat",
        "cat_03.jpg": "cat",
        "cat_04.jpg": "cat",
        "cat_05.jpg": "cat",
        "cat_06.jpg": "cat",
        "cat_07.jpg": "cat",
        "dog_01.jpg": "dog",
        "dog_02.jpg": "dog",
        "dog_03.jpg": "dog",
        "dog_04.jpg": "dog",
        "dog_05.jpg": "dog",
        "dog_06.jpg": "dog",
        "bird_01.jpg": "bird",
        "bird_02.jpg": "bird",
        "bird_03.jpg": "bird",
        "bird_04.jpg": "bird",
        "bird_05.jpg": "bird",
        "bird_06.jpg": "bird",
        "fish_01.jpg": "fish",
        "fish_02.jpg": "fish",
        "fish_03.jpg": "fish",
        "fish_04.jpg": "fish",
        "fish_05.jpg": "fish",
        "fish_06.jpg": "fish",
    }

    # 4. Set your value for k
    k_value = 3

    # 5. Get the labels of the top k predictions
    y_pred_top_k_example = get_top_k_predictions(
        search_results_example, image_labels_mapping, k=k_value
    )

    # y_pred_top_k_example will be:
    # [
    #   ['cat', 'dog', 'cat'],      # For query 1 ('cat') -> correct
    #   ['cat', 'dog', 'dog'],      # For query 2 ('dog') -> correct
    #   ['bird', 'fish', 'cat'],    # For query 3 ('bird') -> correct
    #   ['dog', 'fish', 'cat'],     # For query 4 ('cat') -> correct
    #   ['cat', 'dog', 'bird']      # For query 5 ('fish') -> incorrect
    # ]

    # 6. Calculate top-k accuracy
    accuracy = top_k_accuracy(y_true_example, y_pred_top_k_example)

    print(f"Ground truth labels: {y_true_example}")
    print(f"Top-{k_value} predicted labels: {y_pred_top_k_example}")
    print(f"Top-{k_value} Accuracy: {accuracy:.2f}")  # Expected: 4/5 = 0.80

    # --- Another example for k=1 (standard accuracy) ---
    k_value_1 = 1
    y_pred_top_1_example = get_top_k_predictions(
        search_results_example, image_labels_mapping, k=k_value_1
    )
    accuracy_1 = top_k_accuracy(y_true_example, y_pred_top_1_example)
    print(f"\nTop-{k_value_1} Accuracy: {accuracy_1:.2f}")  # Expected: 2/5 = 0.40
