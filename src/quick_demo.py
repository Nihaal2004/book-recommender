"""
Quick demo script using a sample of data for faster execution
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_all_data
from data_cleaner import (
    clean_books, clean_ratings, clean_to_read, 
    clean_tags, clean_book_tags, validate_join_keys
)
from dataset_preparation import prepare_datasets
from content_based_recommender import ContentBasedRecommender
from collaborative_filtering_recommender import ItemKNNRecommender
from hybrid_recommender import HybridRecommender
from evaluator import create_train_test_split, compare_models
import pickle


def main():
    """Run a quick demo pipeline with sampled data"""
    
    print("="*70)
    print("BOOK RECOMMENDATION SYSTEM - QUICK DEMO")
    print("="*70)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    books, ratings, to_read, tags, book_tags = load_all_data('data')
    
    # Sample data for faster processing
    print("\n[SAMPLING] Taking sample for quick demo...")
    sample_users = ratings['user_id'].unique()[:1000]  # First 1000 users
    ratings = ratings[ratings['user_id'].isin(sample_users)]
    to_read = to_read[to_read['user_id'].isin(sample_users)]
    
    print(f"Sampled: {len(sample_users)} users, {len(ratings)} ratings, {len(to_read)} to_read")
    
    # Step 2: Clean data
    print("\n[STEP 2] Cleaning data...")
    books_clean = clean_books(books)
    ratings_clean = clean_ratings(ratings)
    to_read_clean = clean_to_read(to_read)
    tags_clean = clean_tags(tags)
    book_tags_clean = clean_book_tags(book_tags)
    
    # Step 3: Validate join keys
    print("\n[STEP 3] Validating join keys...")
    ratings_valid, to_read_valid, book_tags_valid = validate_join_keys(
        books_clean, ratings_clean, to_read_clean, book_tags_clean
    )
    
    # Step 4: Prepare datasets
    print("\n[STEP 4] Preparing datasets...")
    interactions, items = prepare_datasets(
        books_clean, ratings_valid, to_read_valid, 
        book_tags_valid, tags_clean, output_dir='outputs'
    )
    
    # Step 5: Create train/test split
    print("\n[STEP 5] Creating train/test split...")
    train_interactions, test_interactions = create_train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    
    # Step 6: Build recommenders
    print("\n[STEP 6] Building recommenders...")
    
    print("\n--- Content-Based Recommender ---")
    content_recommender = ContentBasedRecommender(items)
    
    print("\n--- Collaborative Filtering Recommender ---")
    cf_recommender = ItemKNNRecommender(train_interactions, items)
    
    print("\n--- Hybrid Recommender ---")
    hybrid_recommender = HybridRecommender(content_recommender, cf_recommender)
    
    # Step 7: Evaluate models
    print("\n[STEP 7] Evaluating models...")
    results_df = compare_models(
        content_recommender, cf_recommender, hybrid_recommender,
        train_interactions, test_interactions,
        k_values=[5, 10],
        alpha_values=[0.3, 0.5, 0.7]
    )
    
    # Save results
    results_df.to_csv('outputs/evaluation_results.csv', index=False)
    print("\nEvaluation results saved to outputs/evaluation_results.csv")
    
    # Step 8: Save models
    print("\n[STEP 8] Saving models...")
    with open('outputs/content_recommender.pkl', 'wb') as f:
        pickle.dump(content_recommender, f)
    with open('outputs/cf_recommender.pkl', 'wb') as f:
        pickle.dump(cf_recommender, f)
    with open('outputs/hybrid_recommender.pkl', 'wb') as f:
        pickle.dump(hybrid_recommender, f)
    print("Models saved to outputs/")
    
    # Step 9: Generate sample recommendations
    print("\n[STEP 9] Generating sample recommendations...")
    sample_user_id = interactions['user_id'].iloc[0]
    
    print(f"\nSample recommendations for user {sample_user_id}:")
    print(f"User has {len(interactions[interactions['user_id'] == sample_user_id])} interactions")
    
    print("\n--- Content-Based ---")
    content_recs = content_recommender.recommend_for_user(
        sample_user_id, interactions, top_n=5
    )
    if len(content_recs) > 0:
        print(content_recs[['title', 'authors', 'score']].to_string(index=False))
    else:
        print("No recommendations")
    
    print("\n--- Collaborative Filtering ---")
    cf_recs = cf_recommender.recommend_for_user(sample_user_id, top_n=5)
    if len(cf_recs) > 0:
        print(cf_recs[['title', 'authors', 'score']].to_string(index=False))
    else:
        print("No recommendations")
    
    print("\n--- Hybrid (α=0.5) ---")
    hybrid_recs = hybrid_recommender.recommend_for_user(
        sample_user_id, interactions, top_n=5, alpha=0.5
    )
    if len(hybrid_recs) > 0:
        print(hybrid_recs[['title', 'authors', 'score']].to_string(index=False))
    else:
        print("No recommendations")
    
    print("\n" + "="*70)
    print("DEMO PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Find best model
    best_result = results_df.loc[results_df['precision'].idxmax()]
    print(f"\nBest performing model: {best_result['method']} (K={best_result['k']})")
    print(f"Precision@{best_result['k']}: {best_result['precision']:.4f}")
    print(f"Recall@{best_result['k']}: {best_result['recall']:.4f}")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("1. Run: streamlit run app/app.py")
    print("2. Open browser at http://localhost:8501")
    print("="*70)


if __name__ == '__main__':
    main()
