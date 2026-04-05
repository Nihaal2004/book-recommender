"""
Dataset preparation module for creating interactions and items datasets
"""
import pandas as pd
import os


def create_interactions_dataset(ratings, to_read, output_dir='outputs'):
    """
    Create interactions_clean.csv combining ratings and to_read
    - ratings = explicit feedback with strength 1.0
    - to_read = implicit feedback with strength 0.5
    - remove duplicates by taking max strength
    """
    print("\nCreating interactions dataset...")
    
    # Create ratings interactions (explicit feedback)
    ratings_interactions = ratings[['user_id', 'book_id']].copy()
    ratings_interactions['strength'] = 1.0
    ratings_interactions['type'] = 'rating'
    
    # Create to_read interactions (implicit feedback)
    to_read_interactions = to_read[['user_id', 'book_id']].copy()
    to_read_interactions['strength'] = 0.5
    to_read_interactions['type'] = 'to_read'
    
    # Combine both
    interactions = pd.concat([ratings_interactions, to_read_interactions], ignore_index=True)
    
    print(f"Total interactions before deduplication: {len(interactions)}")
    
    # Remove duplicates by keeping max strength
    interactions = interactions.sort_values('strength', ascending=False)
    interactions = interactions.drop_duplicates(subset=['user_id', 'book_id'], keep='first')
    
    print(f"Total interactions after deduplication: {len(interactions)}")
    print(f"Unique users: {interactions['user_id'].nunique()}")
    print(f"Unique books: {interactions['book_id'].nunique()}")
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'interactions_clean.csv')
    interactions.to_csv(output_path, index=False)
    print(f"Saved interactions to {output_path}")
    
    return interactions


def create_items_dataset(books, book_tags, tags, output_dir='outputs'):
    """
    Create items_clean.csv with book metadata and aggregated tags
    - include book_id, title, authors, and useful metadata
    - create content_text combining title, authors, and important tags
    """
    print("\nCreating items dataset...")
    
    # Merge book_tags with tags to get tag names
    book_tags_with_names = book_tags.merge(
        tags[['tag_id', 'tag_name']], 
        on='tag_id', 
        how='left'
    )
    
    # Aggregate top tags per book (top 10 by count)
    book_tags_agg = book_tags_with_names.sort_values(['goodreads_book_id', 'count'], ascending=[True, False])
    book_tags_agg = book_tags_agg.groupby('goodreads_book_id').head(10)
    
    # Create aggregated tags string
    tags_aggregated = book_tags_agg.groupby('goodreads_book_id')['tag_name'].apply(
        lambda x: ', '.join(x.astype(str))
    ).reset_index()
    tags_aggregated.columns = ['goodreads_book_id', 'tags']
    
    # Merge with books
    items = books.merge(tags_aggregated, on='goodreads_book_id', how='left')
    items['tags'] = items['tags'].fillna('')
    
    # Select useful columns
    items_clean = items[[
        'book_id', 'goodreads_book_id', 'title', 'authors', 
        'original_publication_year', 'language_code', 
        'average_rating', 'ratings_count', 'tags'
    ]].copy()
    
    # Create content_text for content-based recommendations
    items_clean['content_text'] = (
        items_clean['title'].fillna('') + ' ' + 
        items_clean['authors'].fillna('') + ' ' + 
        items_clean['tags'].fillna('')
    )
    
    items_clean['content_text'] = items_clean['content_text'].str.strip()
    
    print(f"Total items: {len(items_clean)}")
    print(f"Items with tags: {(items_clean['tags'] != '').sum()}")
    
    # Save to file
    output_path = os.path.join(output_dir, 'items_clean.csv')
    items_clean.to_csv(output_path, index=False)
    print(f"Saved items to {output_path}")
    
    return items_clean


def prepare_datasets(books, ratings, to_read, book_tags, tags, output_dir='outputs'):
    """Prepare both interactions and items datasets"""
    interactions = create_interactions_dataset(ratings, to_read, output_dir)
    items = create_items_dataset(books, book_tags, tags, output_dir)
    return interactions, items
