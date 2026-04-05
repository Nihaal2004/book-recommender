"""
Content-based recommendation engine using TF-IDF
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based recommender using TF-IDF on book content text
    """
    
    def __init__(self, items_df):
        """
        Initialize with items dataset
        
        Parameters:
        -----------
        items_df : DataFrame with columns ['book_id', 'content_text']
        """
        self.items = items_df.copy()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.book_id_to_idx = {}
        self.idx_to_book_id = {}
        
        self._build_model()
    
    def _build_model(self):
        """Build TF-IDF model and compute similarity matrix"""
        print("Building content-based model...")
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.tfidf_matrix = tfidf.fit_transform(self.items['content_text'].fillna(''))
        
        # Compute similarity matrix (book-to-book)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create mappings
        self.book_id_to_idx = {book_id: idx for idx, book_id in enumerate(self.items['book_id'])}
        self.idx_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_idx.items()}
        
        print(f"Built content-based model with {self.tfidf_matrix.shape[0]} books")
    
    def get_similar_books(self, book_id, top_n=10):
        """Get similar books based on content"""
        if book_id not in self.book_id_to_idx:
            return []
        
        idx = self.book_id_to_idx[book_id]
        similarity_scores = self.similarity_matrix[idx]
        
        # Get top similar books (excluding itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        similar_book_ids = [self.idx_to_book_id[i] for i in similar_indices]
        similar_scores = similarity_scores[similar_indices]
        
        return list(zip(similar_book_ids, similar_scores))
    
    def recommend(self, user_interactions, top_n=10):
        """
        Recommend books for a user based on their interaction history
        
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
            similar_books = self.get_similar_books(book_id, top_n=50)
            
            for similar_book_id, similarity in similar_books:
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
    
    def recommend_for_user(self, user_id, interactions_df, top_n=10):
        """
        Convenience method to recommend for a user given interactions DataFrame
        
        Parameters:
        -----------
        user_id : int
        interactions_df : DataFrame with columns ['user_id', 'book_id', 'strength']
        top_n : number of recommendations
        
        Returns:
        --------
        DataFrame with columns ['book_id', 'score', 'title', 'authors']
        """
        # Get user interactions
        user_data = interactions_df[interactions_df['user_id'] == user_id]
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
