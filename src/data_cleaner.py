"""
Data cleaning module for Goodbooks-10k dataset
"""
import pandas as pd
import numpy as np
import ftfy


def clean_column_names(df):
    """Strip whitespace from column names"""
    df.columns = df.columns.str.strip()
    return df


def fix_encoding(text):
    """Fix mojibake/encoding issues using ftfy"""
    if pd.isna(text):
        return text
    try:
        return ftfy.fix_text(str(text))
    except:
        return text


def clean_books(books):
    """Clean books dataset"""
    books = clean_column_names(books)
    
    # Fix encoding issues in title and authors
    print("Fixing encoding issues...")
    books['title'] = books['title'].apply(fix_encoding)
    books['authors'] = books['authors'].apply(fix_encoding)
    if 'original_title' in books.columns:
        books['original_title'] = books['original_title'].apply(fix_encoding)
    
    # Handle missing values
    books['authors'] = books['authors'].fillna('Unknown Author')
    books['original_title'] = books['original_title'].fillna(books['title'])
    books['language_code'] = books['language_code'].fillna('en')
    
    # Ensure IDs are numeric
    books['book_id'] = pd.to_numeric(books['book_id'], errors='coerce')
    books['goodreads_book_id'] = pd.to_numeric(books['goodreads_book_id'], errors='coerce')
    books['best_book_id'] = pd.to_numeric(books['best_book_id'], errors='coerce')
    
    # Drop rows with missing book_id
    books = books.dropna(subset=['book_id'])
    books['book_id'] = books['book_id'].astype(int)
    
    print(f"Cleaned books: {len(books)} rows")
    return books


def clean_ratings(ratings):
    """Clean ratings dataset"""
    ratings = clean_column_names(ratings)
    
    # Ensure IDs are numeric
    ratings['user_id'] = pd.to_numeric(ratings['user_id'], errors='coerce')
    ratings['book_id'] = pd.to_numeric(ratings['book_id'], errors='coerce')
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
    
    # Drop rows with missing values
    ratings = ratings.dropna()
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['book_id'] = ratings['book_id'].astype(int)
    
    print(f"Cleaned ratings: {len(ratings)} rows")
    return ratings


def clean_to_read(to_read):
    """Clean to_read dataset"""
    to_read = clean_column_names(to_read)
    
    # Ensure IDs are numeric
    to_read['user_id'] = pd.to_numeric(to_read['user_id'], errors='coerce')
    to_read['book_id'] = pd.to_numeric(to_read['book_id'], errors='coerce')
    
    # Drop rows with missing values
    to_read = to_read.dropna()
    to_read['user_id'] = to_read['user_id'].astype(int)
    to_read['book_id'] = to_read['book_id'].astype(int)
    
    print(f"Cleaned to_read: {len(to_read)} rows")
    return to_read


def clean_tags(tags):
    """Clean tags dataset"""
    tags = clean_column_names(tags)
    
    # Fix encoding in tag_name
    print("Fixing tag encoding...")
    tags['tag_name'] = tags['tag_name'].apply(fix_encoding)
    
    # Ensure tag_id is numeric
    tags['tag_id'] = pd.to_numeric(tags['tag_id'], errors='coerce')
    tags = tags.dropna()
    tags['tag_id'] = tags['tag_id'].astype(int)
    
    print(f"Cleaned tags: {len(tags)} rows")
    return tags


def clean_book_tags(book_tags):
    """Clean book_tags dataset"""
    book_tags = clean_column_names(book_tags)
    
    # Ensure IDs are numeric
    book_tags['goodreads_book_id'] = pd.to_numeric(book_tags['goodreads_book_id'], errors='coerce')
    book_tags['tag_id'] = pd.to_numeric(book_tags['tag_id'], errors='coerce')
    book_tags['count'] = pd.to_numeric(book_tags['count'], errors='coerce')
    
    # Drop rows with missing values
    book_tags = book_tags.dropna()
    book_tags['goodreads_book_id'] = book_tags['goodreads_book_id'].astype(int)
    book_tags['tag_id'] = book_tags['tag_id'].astype(int)
    book_tags['count'] = book_tags['count'].astype(int)
    
    print(f"Cleaned book_tags: {len(book_tags)} rows")
    return book_tags


def validate_join_keys(books, ratings, to_read, book_tags):
    """Validate that join keys exist before merging"""
    valid_book_ids = set(books['book_id'].unique())
    valid_goodreads_ids = set(books['goodreads_book_id'].unique())
    
    # Validate ratings
    ratings_valid = ratings[ratings['book_id'].isin(valid_book_ids)]
    print(f"Ratings: {len(ratings)} -> {len(ratings_valid)} valid")
    
    # Validate to_read
    to_read_valid = to_read[to_read['book_id'].isin(valid_book_ids)]
    print(f"To-read: {len(to_read)} -> {len(to_read_valid)} valid")
    
    # Validate book_tags
    book_tags_valid = book_tags[book_tags['goodreads_book_id'].isin(valid_goodreads_ids)]
    print(f"Book-tags: {len(book_tags)} -> {len(book_tags_valid)} valid")
    
    return ratings_valid, to_read_valid, book_tags_valid
