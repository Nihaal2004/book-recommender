# Git Commit Workflow

This document outlines the recommended commit sequence for this project.

## Commit Sequence

### 1. Initial project setup
```bash
git add .gitignore requirements.txt README.md
git commit -m "Initial project setup with dependencies and documentation"
```

### 2. Data loading and cleaning
```bash
git add src/data_loader.py src/data_cleaner.py
git commit -m "Add data loading and cleaning modules"
```

### 3. Interaction and item dataset generation
```bash
git add src/dataset_preparation.py
git commit -m "Add dataset preparation for interactions and items"
```

### 4. Content-based recommender
```bash
git add src/content_based_recommender.py
git commit -m "Implement content-based recommender using TF-IDF"
```

### 5. Collaborative filtering recommender
```bash
git add src/collaborative_filtering_recommender.py
git commit -m "Implement item-item collaborative filtering recommender"
```

### 6. Hybrid recommender
```bash
git add src/hybrid_recommender.py
git commit -m "Implement hybrid recommender combining content and CF"
```

### 7. Evaluation pipeline
```bash
git add src/evaluator.py
git commit -m "Add evaluation framework with Precision@K and Recall@K"
```

### 8. Main pipeline script
```bash
git add src/main.py
git commit -m "Add main pipeline script to orchestrate entire workflow"
```

### 9. Web app UI
```bash
git add app/app.py
git commit -m "Add Streamlit web application with interactive demo"
```

### 10. Final cleanup and documentation
```bash
git add outputs/evaluation_results.csv
git commit -m "Add evaluation results and final documentation updates"
```

## Complete Initial Commit (Alternative)

If you prefer a single initial commit after completing the project:

```bash
git add .
git commit -m "Complete book recommendation system with content-based, CF, and hybrid models"
```

## Pushing to GitHub

```bash
git remote add origin <your-repository-url>
git branch -M main
git push -u origin main
```

## Important Notes

- Large files (CSV data, PKL models) are excluded via `.gitignore`
- Only `evaluation_results.csv` is tracked for demonstration
- Raw data files should be obtained separately and placed in `data/` directory
- Run `python src/main.py` to generate models and outputs before running the app
