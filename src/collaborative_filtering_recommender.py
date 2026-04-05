"""
Item-item collaborative filtering recommender
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNNRecommender:
    """
    Item-item collaborative filtering using cosine similarity
    """
    
    def __init__(self, interactions_df, items_df):
        """
        Initialize with interactions and items datasets
        
        Parameters:
        -----------
        interactions_df : DataFrame with columns ['user_id', 'book_id', 'strength']
        items_df : DataFrame with columns ['book_id', 'title', 'authors']
        """
        self.interactions = interactions_df.copy()
        self.items = items_df.copy()
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        
        self._build_model()
    
    def _build_model(self):
        """Build user-item matrix and compute item-item similarity"""
        print("Building collaborative filtering model...")
        
        # Create user and book mappings
        unique_users = sorted(self.interactions['user_id'].unique())
        unique_books = sorted(self.interactions['book_id'].unique())
        
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        
        # Create user-item matrix
        row_indices = self.interactions['user_id'].map(self.user_id_to_idx)
        col_indices = self.interactions['book_id'].map(self.book_id_to_idx)
        data = self.interactions['strength'].values
        
        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_books))
        )
        
        # Compute item-item similarity
        # Transpose to get item-user matrix, then compute similarity
        item_user_matrix = self.user_item_matrix.T.tocsr()
        self.item_similarity_matrix = cosine_similarity(item_user_matrix, dense_output=False)
        
        print(f"Built CF model: {len(unique_users)} users, {len(unique_books)} books")
        print(f"Sparsity: {100 * (1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])):.2f}%")
    
    def get_similar_items(self, book_id, top_n=10):
        """Get similar items based on collaborative filtering"""
        if book_id not in self.book_id_to_idx:
            return []
        
        idx = self.book_id_to_idx[book_id]
        similarity_scores = self.item_similarity_matrix[idx].toarray().flatten()
        
        # Get top similar items (excluding itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        similar_book_ids = [self.idx_to_book_id[i] for i in similar_indices if i in self.idx_to_book_id]
        similar_scores = similarity_scores[similar_indices]
        
        return list(zip(similar_book_ids, similar_scores))
    
    def recommend(self, user_interactions, top_n=10):
        """
        Recommend books for a user based on item-item similarity
        
        Parameters:
        -----------
        user_interactions : list of (book_id, strength) tuples
        top_n : number of recommendations
        
        Returns:
        --------
        list of (book_id, score) tuples
        """
        if not user_interactions:
            return []
        
        # Aggregate scores from all user interactions
        recommendation_scores = {}
        
        for book_id, strength in user_interactions:
            similar_items = self.get_similar_items(book_id, top_n=50)
            
            for similar_book_id, similarity in similar_items:
                if similar_book_id not in [b for b, _ in user_interactions]:
                    if similar_book_id not in recommendation_scores:
                        recommendation_scores[similar_book_id] = 0
                    recommendation_scores[similar_book_id] += similarity * strength
        
        # Sort and return top N
        recommendations = sorted(
            recommendation_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return recommendations
    
    def recommend_for_user(self, user_id, top_n=10):
        """
        Recommend books for a specific user
        
        Parameters:
        -----------
        user_id : int
        top_n : number of recommendations
        
        Returns:
        --------
        DataFrame with columns ['book_id', 'score', 'title', 'authors']
        """
        # Get user interactions
        user_data = self.interactions[self.interactions['user_id'] == user_id]
        user_interactions = list(zip(user_data['book_id'], user_data['strength']))
        
        # Get recommendations
        recommendations = self.recommend(user_interactions, top_n)
        
        if not recommendations:
            return pd.DataFrame(columns=['book_id', 'score', 'title', 'authors'])
        
        # Create result DataFrame
        rec_df = pd.DataFrame(recommendations, columns=['book_id', 'score'])
        rec_df = rec_df.merge(
            self.items[['book_id', 'title', 'authors']], 
            on='book_id', 
            how='left'
        )
        
        return rec_df
