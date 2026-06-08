"""
Evaluation utilities for H&M Personalized Fashion Recommendations.

Usage:
    from src.eval import map_at_k

    score = map_at_k(predictions, ground_truth, k=12)

Where:
    predictions:  dict  {customer_id: [ordered list of article_ids]}
    ground_truth: dict  {customer_id: {set of actual article_ids}}
"""

import numpy as np


def average_precision_at_k(predicted: list, actual: set, k: int = 12) -> float:
    """
    Compute Average Precision at k for a single customer.

    Args:
        predicted: ordered list of predicted article_ids (length <= k)
        actual:    set of article_ids the customer actually bought
        k:         cutoff (12 for this competition)

    Returns:
        AP@k score (float between 0 and 1)
    """
    if not actual:
        return 0.0

    predicted = predicted[:k]
    hits = 0
    sum_precision = 0.0

    for i, pred in enumerate(predicted):
        if pred in actual:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(len(actual), k)


def map_at_k(predictions: dict, ground_truth: dict, k: int = 12) -> float:
    """
    Compute Mean Average Precision at k across all customers.

    Args:
        predictions:  dict of {customer_id: [list of predicted article_ids]}
        ground_truth: dict of {customer_id: {set of actual article_ids}}
        k:            cutoff

    Returns:
        MAP@k score (float between 0 and 1)
    """
    ap_scores = []

    for customer_id, actual in ground_truth.items():
        predicted = predictions.get(customer_id, [])
        ap_scores.append(average_precision_at_k(predicted, actual, k))

    return np.mean(ap_scores) if ap_scores else 0.0
