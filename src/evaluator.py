"""
Evaluation module for recommendation models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_train_test_split(interactions_df, test_size=0.2, random_state=42):
    """
    Create train/test split by user
    For each user, hold out test_size of their interactions
    """
    print(f"\nCreating train/test split (test_size={test_size})...")
    
    train_data = []
    test_data = []
    
    for user_id in interactions_df['user_id'].unique():
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_interactions) < 2:
            # If user has only 1 interaction, put in train
            train_data.append(user_interactions)
        else:
            train, test = train_test_split(
                user_interactions, 
                test_size=test_size, 
                random_state=random_state
            )
            train_data.append(train)
            test_data.append(test)
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"Train: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
    print(f"Test: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")
    
    return train_df, test_df


def precision_at_k(recommended_items, relevant_items, k):
    """Calculate Precision@K"""
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_k)
    
    if len(recommended_k) == 0:
        return 0.0
    
    return len(recommended_set & relevant_set) / len(recommended_k)


def recall_at_k(recommended_items, relevant_items, k):
    """Calculate Recall@K"""
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recommended_set = set(recommended_k)
    
    if len(relevant_set) == 0:
        return 0.0
    
    return len(recommended_set & relevant_set) / len(relevant_set)


def evaluate_recommender(recommender, train_df, test_df, k=10, method='content', alpha=0.5):
    """
    Evaluate a recommender on test data
    
    Parameters:
    -----------
    recommender : recommender instance
    train_df : training interactions DataFrame
    test_df : test interactions DataFrame
    k : top-K for evaluation
    method : 'content', 'cf', or 'hybrid' (or variation like 'hybrid(α=0.5)')
    alpha : only used for hybrid method
    
    Returns:
    --------
    dict with precision@k and recall@k
    """
    print(f"\nEvaluating {method} recommender (K={k})...")
    
    precision_scores = []
    recall_scores = []
    
    # Get unique users from test set
    test_users = test_df['user_id'].unique()
    
    for user_id in test_users:
        # Get user's train interactions
        user_train = train_df[train_df['user_id'] == user_id]
        
        if len(user_train) == 0:
            continue
        
        # Get user's test interactions (ground truth)
        user_test = test_df[test_df['user_id'] == user_id]
        relevant_items = user_test['book_id'].tolist()
        
        # Get recommendations
        user_interactions = list(zip(user_train['book_id'], user_train['strength']))
        
        if method == 'content':
            recommendations = recommender.recommend(user_interactions, top_n=k)
        elif method == 'cf':
            recommendations = recommender.recommend(user_interactions, top_n=k)
        elif method.startswith('hybrid'):
            recommendations = recommender.recommend(user_interactions, top_n=k, alpha=alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        recommended_items = [book_id for book_id, _ in recommendations]
        
        # Calculate metrics
        precision = precision_at_k(recommended_items, relevant_items, k)
        recall = recall_at_k(recommended_items, relevant_items, k)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"Evaluated on {len(precision_scores)} users")
    
    return {
        'method': method,
        'k': k,
        'precision': avg_precision,
        'recall': avg_recall,
        'num_users': len(precision_scores)
    }


def compare_models(content_recommender, cf_recommender, hybrid_recommender, 
                   train_df, test_df, k_values=[5, 10], alpha_values=[0.3, 0.5, 0.7]):
    """
    Compare all models and find best performing configuration
    
    Returns:
    --------
    DataFrame with comparison results
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results = []
    
    # Evaluate content-based
    for k in k_values:
        result = evaluate_recommender(content_recommender, train_df, test_df, k=k, method='content')
        results.append(result)
    
    # Evaluate CF
    for k in k_values:
        result = evaluate_recommender(cf_recommender, train_df, test_df, k=k, method='cf')
        results.append(result)
    
    # Evaluate hybrid with different alpha values
    for k in k_values:
        for alpha in alpha_values:
            result = evaluate_recommender(
                hybrid_recommender, train_df, test_df, 
                k=k, method=f'hybrid(α={alpha})', alpha=alpha
            )
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df
