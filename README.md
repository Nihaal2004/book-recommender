# 📚 Book Recommendation System

A comprehensive book recommendation system built on the **Goodbooks-10k dataset**, implementing and comparing three recommendation approaches: content-based filtering, collaborative filtering, and a hybrid model.

## 🎯 Project Overview

This project demonstrates a clean, explainable recommendation pipeline with:
- **Data cleaning and preparation** from multiple data sources
- **Three recommendation models**: content-based, item-item collaborative filtering, and hybrid
- **Evaluation framework** using Precision@K and Recall@K metrics
- **Interactive web application** built with Streamlit for demonstration

## 📊 Dataset

The **Goodbooks-10k dataset** includes:
- **10,000 books** with metadata (titles, authors, ratings, tags)
- **~6M ratings** (explicit feedback)
- **~900K to-read entries** (implicit feedback)
- **User-generated tags** describing book content

### Data Files
- `books.csv` - Book metadata
- `ratings.csv` - User ratings (1-5 stars)
- `to_read.csv` - User reading lists
- `tags.csv` - Tag definitions
- `book_tags.csv` - Book-tag associations

## 🚀 Methods Implemented

### 1. Content-Based Filtering
- Uses **TF-IDF** on book titles, authors, and tags
- Computes **cosine similarity** between books
- Recommends books similar to user's reading history

### 2. Collaborative Filtering (Item-Item)
- Builds **user-item interaction matrix**
- Computes **item-item cosine similarity**
- Recommends books popular among similar users

### 3. Hybrid Model
- Combines content-based and collaborative filtering
- Uses tunable parameter **α** to balance approaches
  - `α = 0.0` → Pure collaborative filtering
  - `α = 1.0` → Pure content-based
  - `α = 0.5` → Equal weight
- Leverages strengths of both methods

## 📁 Project Structure

```
predictive/
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   ├── books.csv
│   ├── ratings.csv
│   ├── to_read.csv
│   ├── tags.csv
│   └── book_tags.csv
├── src/
│   ├── data_loader.py                        # Data loading utilities
│   ├── data_cleaner.py                       # Data cleaning functions
│   ├── dataset_preparation.py                # Dataset preparation
│   ├── content_based_recommender.py          # Content-based model
│   ├── collaborative_filtering_recommender.py # CF model
│   ├── hybrid_recommender.py                 # Hybrid model
│   ├── evaluator.py                          # Evaluation framework
│   └── main.py                               # Main pipeline script
├── outputs/
│   ├── interactions_clean.csv    # Cleaned interactions dataset
│   ├── items_clean.csv          # Cleaned items dataset
│   ├── evaluation_results.csv   # Model evaluation results
│   └── *.pkl                    # Saved models
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd predictive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 How to Run

### 1. Run the Pipeline

Execute the main pipeline to clean data, build models, and evaluate:

```bash
python src/main.py
```

This will:
- Load and clean the raw data
- Create interactions and items datasets
- Build all three recommender models
- Evaluate models on test data
- Save models and results to `outputs/`

### 2. Launch the Web App

Start the Streamlit web application:

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Web Application Features

The web app includes five main sections:

### 1. **Home**
- Project overview
- Dataset statistics
- Explanation of recommendation methods

### 2. **Data Pipeline**
- Visual explanation of data processing steps
- Sample data from cleaned datasets
- Dataset statistics

### 3. **Recommendations**
- Interactive recommendation demo
- Select user, method, and parameters
- View user's reading history
- Get personalized recommendations

### 4. **Model Comparison**
- Evaluation results table
- Visual comparison charts
- Performance metrics (Precision@K, Recall@K)
- Best model identification

### 5. **Explainability**
- Explanation of why books are recommended
- Show influencing books from user's history
- Content and collaborative similarity breakdown

## 📈 Evaluation Results

Models are evaluated using:
- **Holdout validation** (80% train, 20% test per user)
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant items in top-K

Sample results (will vary based on actual run):
```
Method              K  Precision  Recall
Content-Based       10   0.0234    0.0421
Collaborative       10   0.0312    0.0587
Hybrid (α=0.5)      10   0.0289    0.0534
```

## 🎨 Sample Output

### User Reading History
```
Title                          Authors              Rating  Type
Harry Potter and the Sorcer... J.K. Rowling         4.47    rating
The Hobbit                     J.R.R. Tolkien       4.26    rating
1984                          George Orwell         4.17    to_read
```

### Recommendations (Hybrid, α=0.5)
```
1. The Lord of the Rings       J.R.R. Tolkien       Score: 0.842
2. The Chronicles of Narnia    C.S. Lewis           Score: 0.789
3. Animal Farm                 George Orwell        Score: 0.756
```

## 🔑 Key Findings

- **Collaborative filtering** generally performs best on this dataset due to rich user interaction data
- **Content-based** is useful for cold-start scenarios and niche genres
- **Hybrid models** with `α ≈ 0.3-0.5` provide balanced recommendations
- **Explainability** features help users understand recommendation reasoning

## 📝 Development Workflow

The project follows a clean commit history:

1. Initial project setup
2. Data loading and cleaning
3. Interaction and item dataset generation
4. Content-based recommender
5. Collaborative filtering recommender
6. Hybrid recommender
7. Evaluation pipeline
8. Web app UI
9. Model comparison and explainability
10. Final cleanup and documentation

## 🤝 Contributing

This is a demonstration project for educational purposes. Feel free to fork and experiment!

## 📄 License

This project is for educational purposes. The Goodbooks-10k dataset is from Kaggle.

## 👨‍💻 Author

Built as a comprehensive recommendation system project demonstrating data pipeline, model development, evaluation, and deployment.

---

**Note**: Large data files (CSV, PKL) are excluded from git via `.gitignore`. Run the pipeline first to generate them.
