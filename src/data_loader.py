"""
Data loading module for Goodbooks-10k dataset
"""
import pandas as pd
import os


def load_books(data_dir='data'):
    """Load books dataset with only necessary columns"""
    books_path = os.path.join(data_dir, 'books.csv')
    books = pd.read_csv(books_path)
    
    # Rename 'id' to 'goodreads_book_id' to match book_tags
    if 'id' in books.columns:
        books = books.rename(columns={'id': 'goodreads_book_id'})
    
    return books


def load_ratings(data_dir='data'):
    """Load ratings dataset"""
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    ratings = pd.read_csv(ratings_path)
    return ratings


def load_to_read(data_dir='data'):
    """Load to_read dataset"""
    to_read_path = os.path.join(data_dir, 'to_read.csv')
    to_read = pd.read_csv(to_read_path)
    return to_read


def load_tags(data_dir='data'):
    """Load tags dataset"""
    tags_path = os.path.join(data_dir, 'tags.csv')
    tags = pd.read_csv(tags_path)
    return tags


def load_book_tags(data_dir='data'):
    """Load book_tags dataset"""
    book_tags_path = os.path.join(data_dir, 'book_tags.csv')
    book_tags = pd.read_csv(book_tags_path)
    return book_tags


def load_all_data(data_dir='data'):
    """Load all datasets at once"""
    print("Loading books...")
    books = load_books(data_dir)
    
    print("Loading ratings...")
    ratings = load_ratings(data_dir)
    
    print("Loading to_read...")
    to_read = load_to_read(data_dir)
    
    print("Loading tags...")
    tags = load_tags(data_dir)
    
    print("Loading book_tags...")
    book_tags = load_book_tags(data_dir)
    
    print(f"Loaded {len(books)} books, {len(ratings)} ratings, "
          f"{len(to_read)} to_read entries, {len(tags)} tags, "
          f"{len(book_tags)} book-tag associations")
    
    return books, ratings, to_read, tags, book_tags
