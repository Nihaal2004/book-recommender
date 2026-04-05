"""
Hybrid recommender combining content-based and collaborative filtering
"""
import pandas as pd
import numpy as np


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering
    """
    
    def __init__(self, content_recommender, cf_recommender):
        """
        Initialize with both recommenders
        
        Parameters:
        -----------
        content_recommender : ContentBasedRecommender instance
        cf_recommender : ItemKNNRecommender instance
        """
        self.content_recommender = content_recommender
        self.cf_recommender = cf_recommender
        self.items = content_recommender.items
    
    def recommend(self, user_interactions, top_n=10, alpha=0.5):
        """
        Generate hybrid recommendations
        
        Parameters:
        -----------
        user_interactions : list of (book_id, strength) tuples
        top_n : number of recommendations
        alpha : weight for content-based (1-alpha for CF)
                alpha=0.0 -> pure CF
                alpha=1.0 -> pure content-based
                alpha=0.5 -> equal weight
        
        Returns:
        --------
        list of (book_id, score) tuples
        """
        if not user_interactions:
            return []
        
        # Get recommendations from both methods
        content_recs = self.content_recommender.recommend(user_interactions, top_n=100)
        cf_recs = self.cf_recommender.recommend(user_interactions, top_n=100)
        
        # Normalize scores to [0, 1] range
        def normalize_scores(recs):
            if not recs:
                return {}
            scores = [score for _, score in recs]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return {book_id: 1.0 for book_id, _ in recs}
            return {
                book_id: (score - min_score) / (max_score - min_score) 
                for book_id, score in recs
            }
        
        content_scores = normalize_scores(content_recs)
        cf_scores = normalize_scores(cf_recs)
        
        # Combine scores
        all_book_ids = set(content_scores.keys()) | set(cf_scores.keys())
        hybrid_scores = {}
        
        for book_id in all_book_ids:
            content_score = content_scores.get(book_id, 0)
            cf_score = cf_scores.get(book_id, 0)
            hybrid_scores[book_id] = alpha * content_score + (1 - alpha) * cf_score
        
        # Sort and return top N
        recommendations = sorted(
            hybrid_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return recommendations
    
    def recommend_for_user(self, user_id, interactions_df, top_n=10, alpha=0.5):
        """
        Recommend books for a specific user
        
        Parameters:
        -----------
        user_id : int
        interactions_df : DataFrame with columns ['user_id', 'book_id', 'strength']
        top_n : number of recommendations
        alpha : weight for content-based recommendations
        
        Returns:
        --------
        DataFrame with columns ['book_id', 'score', 'title', 'authors']
        """
        # Get user interactions
        user_data = interactions_df[interactions_df['user_id'] == user_id]
        user_interactions = list(zip(user_data['book_id'], user_data['strength']))
        
        # Get recommendations
        recommendations = self.recommend(user_interactions, top_n, alpha)
        
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
    
    def get_explanation(self, user_id, interactions_df, recommended_book_id, alpha=0.5):
        """
        Get explanation for why a book was recommended
        
        Returns information about content similarity and CF contribution
        """
        # Get user interactions
        user_data = interactions_df[interactions_df['user_id'] == user_id]
        user_interactions = list(zip(user_data['book_id'], user_data['strength']))
        
        # Get content-based similar books
        content_influences = []
        for book_id, strength in user_interactions:
            similar_books = self.content_recommender.get_similar_books(book_id, top_n=20)
            for sim_book_id, sim_score in similar_books:
                if sim_book_id == recommended_book_id:
                    content_influences.append({
                        'source_book_id': book_id,
                        'similarity': sim_score,
                        'strength': strength
                    })
        
        # Get CF-based similar books
        cf_influences = []
        for book_id, strength in user_interactions:
            similar_books = self.cf_recommender.get_similar_items(book_id, top_n=20)
            for sim_book_id, sim_score in similar_books:
                if sim_book_id == recommended_book_id:
                    cf_influences.append({
                        'source_book_id': book_id,
                        'similarity': sim_score,
                        'strength': strength
                    })
        
        return {
            'content_influences': content_influences,
            'cf_influences': cf_influences,
            'alpha': alpha
        }
