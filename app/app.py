"""
Streamlit Web App for Book Recommendation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from content_based_recommender import ContentBasedRecommender
from collaborative_filtering_recommender import ItemKNNRecommender
from hybrid_recommender import HybridRecommender


# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load datasets"""
    interactions = pd.read_csv('outputs/interactions_clean.csv')
    items = pd.read_csv('outputs/items_clean.csv')
    evaluation_results = pd.read_csv('outputs/evaluation_results.csv')
    return interactions, items, evaluation_results


@st.cache_resource
def load_models():
    """Load trained models"""
    with open('outputs/content_recommender.pkl', 'rb') as f:
        content_recommender = pickle.load(f)
    with open('outputs/cf_recommender.pkl', 'rb') as f:
        cf_recommender = pickle.load(f)
    with open('outputs/hybrid_recommender.pkl', 'rb') as f:
        hybrid_recommender = pickle.load(f)
    return content_recommender, cf_recommender, hybrid_recommender


def main():
    """Main app function"""
    
    # Load data and models
    try:
        interactions, items, evaluation_results = load_data()
        content_recommender, cf_recommender, hybrid_recommender = load_models()
    except Exception as e:
        st.error(f"Error loading data/models: {e}")
        st.info("Please run the main pipeline first: python src/main.py")
        return
    
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Pipeline", "Recommendations", "Model Comparison", "Explainability"]
    )
    
    # ===== HOME PAGE =====
    if page == "Home":
        st.title("📚 Book Recommendation System")
        st.markdown("---")
        
        st.header("Project Overview")
        st.write("""
        This project demonstrates a comprehensive book recommendation system using the **Goodbooks-10k dataset**.
        The system implements and compares three different recommendation approaches to help users discover books 
        they might enjoy based on their reading history.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Books", f"{len(items):,}")
        with col2:
            st.metric("Total Users", f"{interactions['user_id'].nunique():,}")
        with col3:
            st.metric("Total Interactions", f"{len(interactions):,}")
        
        st.markdown("---")
        st.header("Dataset")
        st.write("""
        The **Goodbooks-10k dataset** contains:
        - **Books**: 10,000 popular books with metadata (titles, authors, ratings, tags)
        - **Ratings**: Explicit user ratings (1-5 stars)
        - **To-Read Lists**: Implicit feedback from users' reading lists
        - **Tags**: User-generated tags describing book content and themes
        """)
        
        st.markdown("---")
        st.header("Recommendation Methods")
        
        st.subheader("1. Content-Based Filtering")
        st.write("""
        Recommends books similar to what the user has read before, based on book content:
        - Uses **TF-IDF** on book titles, authors, and tags
        - Computes **cosine similarity** between books
        - Best for users with specific genre preferences
        """)
        
        st.subheader("2. Collaborative Filtering (Item-Item)")
        st.write("""
        Recommends books liked by users with similar reading patterns:
        - Builds a **user-item interaction matrix**
        - Computes **item-item similarity** based on user behavior
        - Discovers books popular among similar readers
        """)
        
        st.subheader("3. Hybrid Model")
        st.write("""
        Combines both content-based and collaborative filtering:
        - Uses a tunable parameter **α** to balance both approaches
        - **α = 0**: Pure collaborative filtering
        - **α = 1**: Pure content-based
        - **α = 0.5**: Equal weight to both methods
        - Leverages strengths of both approaches
        """)
    
    # ===== DATA PIPELINE PAGE =====
    elif page == "Data Pipeline":
        st.title("📊 Data Pipeline")
        st.markdown("---")
        
        st.header("Pipeline Overview")
        st.write("""
        The data preparation pipeline consists of several steps to transform raw data 
        into clean datasets ready for recommendation models.
        """)
        
        st.subheader("Raw Data Files")
        st.write("""
        - `books.csv`: Book metadata
        - `ratings.csv`: User ratings (explicit feedback)
        - `to_read.csv`: User reading lists (implicit feedback)
        - `tags.csv`: Tag definitions
        - `book_tags.csv`: Book-tag associations
        """)
        
        st.markdown("---")
        st.header("Cleaned Datasets")
        
        st.subheader("1. Interactions Dataset")
        st.write("""
        Combines explicit ratings (strength=1.0) and implicit to-read data (strength=0.5).
        Duplicates are removed by keeping the maximum strength value.
        """)
        
        st.dataframe(interactions.head(10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interactions", f"{len(interactions):,}")
        with col2:
            st.metric("Unique Users", f"{interactions['user_id'].nunique():,}")
        with col3:
            st.metric("Unique Books", f"{interactions['book_id'].nunique():,}")
        
        st.markdown("---")
        st.subheader("2. Items Dataset")
        st.write("""
        Contains book metadata with aggregated tags and content text for content-based filtering.
        """)
        
        display_items = items[['book_id', 'title', 'authors', 'average_rating', 'tags']].head(10)
        st.dataframe(display_items)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Books", f"{len(items):,}")
        with col2:
            st.metric("Books with Tags", f"{(items['tags'] != '').sum():,}")
    
    # ===== RECOMMENDATIONS PAGE =====
    elif page == "Recommendations":
        st.title("🎯 Book Recommendations")
        st.markdown("---")
        
        # User selection
        st.sidebar.header("Configuration")
        user_ids = sorted(interactions['user_id'].unique())
        selected_user = st.sidebar.selectbox("Select User ID", user_ids, index=0)
        
        # Method selection
        method = st.sidebar.selectbox(
            "Recommendation Method",
            ["Content-Based", "Collaborative Filtering", "Hybrid"]
        )
        
        # Top N
        top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
        
        # Alpha for hybrid
        alpha = 0.5
        if method == "Hybrid":
            alpha = st.sidebar.slider("Alpha (Content Weight)", 0.0, 1.0, 0.5, 0.1)
            st.sidebar.caption(f"α={alpha:.1f} means {alpha*100:.0f}% content, {(1-alpha)*100:.0f}% collaborative")
        
        # Show user's reading history
        st.header(f"User {selected_user}'s Reading History")
        user_books = interactions[interactions['user_id'] == selected_user].merge(
            items[['book_id', 'title', 'authors', 'average_rating']], 
            on='book_id'
        ).sort_values('strength', ascending=False)
        
        st.write(f"Total books: {len(user_books)}")
        st.dataframe(user_books[['title', 'authors', 'average_rating', 'strength', 'type']].head(10))
        
        st.markdown("---")
        st.header(f"Recommendations ({method})")
        
        # Generate recommendations
        if method == "Content-Based":
            recommendations = content_recommender.recommend_for_user(
                selected_user, interactions, top_n=top_n
            )
        elif method == "Collaborative Filtering":
            recommendations = cf_recommender.recommend_for_user(
                selected_user, top_n=top_n
            )
        else:  # Hybrid
            recommendations = hybrid_recommender.recommend_for_user(
                selected_user, interactions, top_n=top_n, alpha=alpha
            )
        
        if len(recommendations) == 0:
            st.warning("No recommendations available for this user.")
        else:
            # Display recommendations as cards
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"{idx+1}. {row['title']}")
                        st.write(f"**Author(s):** {row['authors']}")
                    with col2:
                        st.metric("Score", f"{row['score']:.3f}")
                    st.markdown("---")
    
    # ===== MODEL COMPARISON PAGE =====
    elif page == "Model Comparison":
        st.title("📈 Model Comparison")
        st.markdown("---")
        
        st.header("Evaluation Results")
        st.write("""
        Models are evaluated using holdout validation with Precision@K and Recall@K metrics.
        20% of each user's interactions are held out for testing.
        """)
        
        # Display results table
        st.subheader("Performance Metrics")
        st.dataframe(evaluation_results)
        
        # Find best model
        best_idx = evaluation_results['precision'].idxmax()
        best_model = evaluation_results.loc[best_idx]
        
        st.success(f"""
        **Best Performing Model:** {best_model['method']} (K={best_model['k']})  
        **Precision@{best_model['k']}:** {best_model['precision']:.4f}  
        **Recall@{best_model['k']}:** {best_model['recall']:.4f}
        """)
        
        st.markdown("---")
        st.header("Visual Comparison")
        
        # Precision comparison chart
        st.subheader("Precision@K Comparison")
        fig_precision = px.bar(
            evaluation_results,
            x='method',
            y='precision',
            color='k',
            barmode='group',
            title='Precision@K by Method',
            labels={'precision': 'Precision', 'method': 'Method', 'k': 'K'}
        )
        st.plotly_chart(fig_precision, use_container_width=True)
        
        # Recall comparison chart
        st.subheader("Recall@K Comparison")
        fig_recall = px.bar(
            evaluation_results,
            x='method',
            y='recall',
            color='k',
            barmode='group',
            title='Recall@K by Method',
            labels={'recall': 'Recall', 'method': 'Method', 'k': 'K'}
        )
        st.plotly_chart(fig_recall, use_container_width=True)
        
        st.markdown("---")
        st.header("Analysis")
        st.write("""
        **Key Insights:**
        
        - **Content-Based** works well for users with clear genre preferences, as it focuses on book similarity.
        - **Collaborative Filtering** discovers diverse recommendations by learning from similar users' behavior.
        - **Hybrid models** typically balance both approaches, offering both familiar and diverse recommendations.
        - The optimal **α** parameter depends on the dataset and user preferences.
        """)
    
    # ===== EXPLAINABILITY PAGE =====
    elif page == "Explainability":
        st.title("🔍 Recommendation Explainability")
        st.markdown("---")
        
        st.write("""
        This page explains why specific books are recommended by showing which books 
        from the user's history influenced each recommendation.
        """)
        
        # User selection
        st.sidebar.header("Configuration")
        user_ids = sorted(interactions['user_id'].unique())
        selected_user = st.sidebar.selectbox("Select User ID", user_ids, index=0)
        
        # Method selection
        method = st.sidebar.selectbox(
            "Recommendation Method",
            ["Content-Based", "Collaborative Filtering", "Hybrid"]
        )
        
        alpha = 0.5
        if method == "Hybrid":
            alpha = st.sidebar.slider("Alpha (Content Weight)", 0.0, 1.0, 0.5, 0.1)
        
        # Generate recommendations
        if method == "Content-Based":
            recommendations = content_recommender.recommend_for_user(
                selected_user, interactions, top_n=5
            )
        elif method == "Collaborative Filtering":
            recommendations = cf_recommender.recommend_for_user(
                selected_user, top_n=5
            )
        else:  # Hybrid
            recommendations = hybrid_recommender.recommend_for_user(
                selected_user, interactions, top_n=5, alpha=alpha
            )
        
        if len(recommendations) == 0:
            st.warning("No recommendations available for this user.")
            return
        
        # Select a recommended book
        rec_titles = [f"{row['title']} - {row['authors']}" for _, row in recommendations.iterrows()]
        selected_rec = st.selectbox("Select a recommended book", rec_titles)
        selected_book_id = recommendations.iloc[rec_titles.index(selected_rec)]['book_id']
        
        st.markdown("---")
        st.header("Why was this book recommended?")
        
        # Get user's reading history
        user_data = interactions[interactions['user_id'] == selected_user]
        user_books_info = user_data.merge(
            items[['book_id', 'title', 'authors']], 
            on='book_id'
        )
        
        if method == "Content-Based":
            st.subheader("Content-Based Explanation")
            st.write("This book is similar to books you've read based on title, authors, and tags.")
            
            # Find similar books from user's history
            user_interactions = list(zip(user_data['book_id'], user_data['strength']))
            influences = []
            
            for book_id, strength in user_interactions:
                similar = content_recommender.get_similar_books(book_id, top_n=20)
                for sim_id, sim_score in similar:
                    if sim_id == selected_book_id:
                        book_info = items[items['book_id'] == book_id].iloc[0]
                        influences.append({
                            'title': book_info['title'],
                            'authors': book_info['authors'],
                            'similarity': sim_score,
                            'strength': strength
                        })
            
            if influences:
                influence_df = pd.DataFrame(influences).sort_values('similarity', ascending=False)
                st.dataframe(influence_df)
            else:
                st.info("No direct content similarity found.")
        
        elif method == "Collaborative Filtering":
            st.subheader("Collaborative Filtering Explanation")
            st.write("Users who read similar books to you also enjoyed this book.")
            
            # Find similar books from user's history
            user_interactions = list(zip(user_data['book_id'], user_data['strength']))
            influences = []
            
            for book_id, strength in user_interactions:
                similar = cf_recommender.get_similar_items(book_id, top_n=20)
                for sim_id, sim_score in similar:
                    if sim_id == selected_book_id:
                        book_info = items[items['book_id'] == book_id].iloc[0]
                        influences.append({
                            'title': book_info['title'],
                            'authors': book_info['authors'],
                            'similarity': sim_score,
                            'strength': strength
                        })
            
            if influences:
                influence_df = pd.DataFrame(influences).sort_values('similarity', ascending=False)
                st.dataframe(influence_df)
            else:
                st.info("No direct collaborative similarity found.")
        
        else:  # Hybrid
            st.subheader("Hybrid Explanation")
            st.write(f"This recommendation combines content similarity (α={alpha:.1f}) and collaborative filtering (α={1-alpha:.1f}).")
            
            # Get explanation from hybrid model
            explanation = hybrid_recommender.get_explanation(
                selected_user, interactions, selected_book_id, alpha
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Content-Based Influences:**")
                if explanation['content_influences']:
                    content_df = pd.DataFrame(explanation['content_influences'])
                    content_df = content_df.merge(
                        items[['book_id', 'title', 'authors']].rename(columns={'book_id': 'source_book_id'}),
                        on='source_book_id'
                    )
                    st.dataframe(content_df[['title', 'authors', 'similarity']])
                else:
                    st.info("No content-based influences")
            
            with col2:
                st.write("**Collaborative Filtering Influences:**")
                if explanation['cf_influences']:
                    cf_df = pd.DataFrame(explanation['cf_influences'])
                    cf_df = cf_df.merge(
                        items[['book_id', 'title', 'authors']].rename(columns={'book_id': 'source_book_id'}),
                        on='source_book_id'
                    )
                    st.dataframe(cf_df[['title', 'authors', 'similarity']])
                else:
                    st.info("No collaborative filtering influences")


if __name__ == '__main__':
    main()
