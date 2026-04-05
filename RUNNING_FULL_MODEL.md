# Running the Full Model vs Quick Demo

## Understanding the Difference

### Quick Demo (quick_demo.py)
- **Users**: 1,000 (sampled from 53,424)
- **Runtime**: 2-3 minutes
- **Purpose**: Fast demonstration and testing
- **Best for**: Project reviews, quick testing, development

### Full Model (main.py)
- **Users**: ALL ~53,424 users
- **Runtime**: 10-20 minutes (depending on your machine)
- **Purpose**: Production-quality evaluation
- **Best for**: Final results, complete analysis, publication-ready metrics

---

## How to Run the FULL MODEL

### Step 1: Install Dependencies (if not already done)
```bash
pip install -r requirements.txt
```

### Step 2: Run the Full Pipeline
```bash
python src/main.py
```

This will:
1. Load ALL data (10,000 books, ~981,756 ratings, ~912,705 to-read entries)
2. Clean and prepare complete datasets
3. Build models on full user base
4. Evaluate on ALL users (not just 1000)
5. Save complete models to `outputs/`
6. Generate comprehensive results

**Expected Runtime:** 10-20 minutes
**Memory Usage:** ~2-4 GB RAM
**Output Files:** Same location (`outputs/`), but with full data

### Step 3: Launch Web App with Full Data
```bash
streamlit run app/app.py
```

The app will automatically load the full models from `outputs/`.

---

## Key Differences in Results

### Quick Demo Results (1,000 users):
```
Method              Precision@5   Recall@5
Content-Based       0.0137        0.0181
Collaborative       0.0969        0.1542
Hybrid (α=0.3)      0.0981        0.1534  ← Best
```

### Full Model Results (ALL users - Expected):
Performance will be **more accurate and generalizable** because:
- More diverse user behavior patterns
- Better collaborative filtering signals
- More robust evaluation
- Production-quality metrics

**Note:** The full model metrics may differ slightly from the demo version due to the larger dataset.

---

## Which Should You Use?

### Use Quick Demo (`python src/quick_demo.py`) if:
- ✓ You want to test the system quickly
- ✓ You're demonstrating in a project review (2-3 minutes)
- ✓ You're developing/debugging code
- ✓ You have limited time or resources

### Use Full Model (`python src/main.py`) if:
- ✓ You want production-quality results
- ✓ You're writing a final report with metrics
- ✓ You need comprehensive evaluation
- ✓ You want to publish or present final findings
- ✓ You have 15-20 minutes to spare

---

## File Sizes Comparison

### Quick Demo Outputs:
- `content_recommender.pkl`: ~800 MB
- `cf_recommender.pkl`: ~2.7 MB
- `hybrid_recommender.pkl`: ~804 MB
- `interactions_clean.csv`: ~268 KB (1,000 users)
- `items_clean.csv`: ~1.6 MB

### Full Model Outputs:
- `content_recommender.pkl`: ~800 MB (same, uses all books)
- `cf_recommender.pkl`: ~50-100 MB (much larger with all users)
- `hybrid_recommender.pkl`: ~850 MB
- `interactions_clean.csv`: ~15-20 MB (all users)
- `items_clean.csv`: ~1.6 MB (same)

---

## Running Both Versions

You can run both and compare:

```bash
# Run quick demo first
python src/quick_demo.py
# Outputs saved to outputs/

# Copy outputs to a backup folder
mkdir outputs_demo
copy outputs\* outputs_demo\

# Run full model
python src/main.py
# Outputs will overwrite with full data

# Now you have both:
# - outputs/ = Full model results
# - outputs_demo/ = Quick demo results
```

---

## Troubleshooting Full Model

### Issue: Out of Memory
**Solution 1**: Close other applications
**Solution 2**: Use quick_demo.py instead
**Solution 3**: Reduce sample size in main.py (edit line to use 5000 users instead)

### Issue: Taking Too Long
**Normal**: 10-20 minutes is expected for full dataset
**Check**: Look for progress messages in console
**Alternative**: Use quick_demo.py for faster results

### Issue: Disk Space
**Required**: ~3-4 GB for all outputs
**Check**: Ensure you have enough space in D:\Projects\predictive\outputs\

---

## Recommendation for Your Project

### For Project Review/Demo:
**Use Quick Demo** - It's fast, sufficient, and demonstrates all features perfectly.

### For Final Report/Metrics:
**Use Full Model** - Provides production-quality, comprehensive results.

### For GitHub:
The evaluation_results.csv from quick demo is already pushed, which is fine for demonstration.

---

## Command Quick Reference

```bash
# Quick Demo (2-3 minutes)
python src/quick_demo.py

# Full Model (10-20 minutes)
python src/main.py

# Launch Web App (works with either)
streamlit run app/app.py

# Check which version is loaded
python -c "import pandas as pd; df=pd.read_csv('outputs/interactions_clean.csv'); print(f'Users: {df.user_id.nunique()}')"
```

---

## Current Status

**What you have now:** Quick demo results (1,000 users)
**What you can run:** Full model anytime with `python src/main.py`
**Both are valid:** Choose based on your needs and time constraints

---

**TLDR:**
- **Quick Demo**: `python src/quick_demo.py` (3 min, 1K users) ← Currently what you have
- **Full Model**: `python src/main.py` (15 min, 53K users) ← Run this for complete results
- **Both work with the same web app**
