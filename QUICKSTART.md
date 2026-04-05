# 🚀 Quick Start Guide

This guide helps you get the Book Recommendation System up and running in minutes.

## ⚡ Fast Track (Recommended for Demo)

### Step 1: Install Dependencies (30 seconds)
```bash
pip install -r requirements.txt
```

### Step 2: Run Quick Demo (2-3 minutes)
```bash
python src/quick_demo.py
```

This will:
- ✓ Load and clean data (sampled: 1000 users)
- ✓ Build all three recommender models
- ✓ Evaluate and compare models
- ✓ Save outputs to `outputs/` directory
- ✓ Display sample recommendations

### Step 3: Launch Web App (5 seconds)
```bash
streamlit run app/app.py
```

Your browser will open to `http://localhost:8501`

**That's it! You're ready to explore.**

---

## 🎯 What You'll See

### Quick Demo Output
```
======================================================================
BOOK RECOMMENDATION SYSTEM - QUICK DEMO
======================================================================

[STEP 1] Loading data...
Loaded 10000 books, 981756 ratings, 912705 to_read entries...

[STEP 7] Evaluating models...
Precision@5: 0.0981 (Hybrid α=0.3) ← Best Model
Recall@5: 0.1534

[STEP 9] Sample recommendations for user 314:
--- Hybrid (α=0.5) ---
1. Harry Potter Boxed Set
2. Harry Potter Collection
3. The Lord of the Rings
...
```

### Web App Pages
1. **Home** - Overview and methodology
2. **Data Pipeline** - Data flow visualization
3. **Recommendations** - Interactive demo
4. **Model Comparison** - Performance charts
5. **Explainability** - Why these books?

---

## 🔧 Full Pipeline (Optional)

For the complete dataset (takes 10-15 minutes):

```bash
python src/main.py
streamlit run app/app.py
```

Uses all users instead of 1000-user sample.

---

## 📊 Expected Results

### Demo Performance (1000 users)
| Model | Precision@5 | Recall@5 |
|-------|-------------|----------|
| Content | 0.014 | 0.018 |
| Collaborative | 0.097 | 0.154 |
| **Hybrid (α=0.3)** | **0.098** | **0.153** |

### Files Created
```
outputs/
├── interactions_clean.csv       (268 KB)
├── items_clean.csv             (1.6 MB)
├── evaluation_results.csv      (621 B)
├── content_recommender.pkl     (803 MB)
├── cf_recommender.pkl          (2.7 MB)
└── hybrid_recommender.pkl      (804 MB)
```

---

## 🎮 Using the Web App

### Get Recommendations
1. Go to **Recommendations** page
2. Select a user ID from dropdown
3. Choose method: Content / CF / Hybrid
4. Adjust parameters (top N, alpha)
5. View personalized book recommendations

### Compare Models
1. Go to **Model Comparison** page
2. See performance table and charts
3. Identify best-performing model

### Understand Recommendations
1. Go to **Explainability** page
2. Select a user and recommended book
3. See which books influenced the recommendation
4. Understand content vs. collaborative contributions

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "File not found" in app
**Solution**: Run the demo first
```bash
python src/quick_demo.py
```

### Issue: Port 8501 already in use
**Solution**: Use different port
```bash
streamlit run app/app.py --server.port 8502
```

### Issue: Memory error with full pipeline
**Solution**: Use quick demo instead
```bash
python src/quick_demo.py  # Uses sampled data
```

---

## 📖 Data Files Required

The following files should be in the `data/` directory:
- ✓ books.csv (3.3 MB)
- ✓ ratings.csv (12 MB)
- ✓ to_read.csv (9.4 MB)
- ✓ tags.csv (722 KB)
- ✓ book_tags.csv (16 MB)

These are already present in your project.

---

## 💡 Tips

### For Project Review
- Run **quick_demo.py** (faster, sufficient for demonstration)
- Focus on **Model Comparison** page in the app
- Show **Explainability** feature (unique selling point)

### For Development
- Run **main.py** for full evaluation
- Experiment with alpha values in hybrid model
- Check `evaluation_results.csv` for detailed metrics

### For Presentation
1. Start with **Home** page (context)
2. Show **Data Pipeline** (methodology)
3. Demo **Recommendations** (live interaction)
4. Present **Model Comparison** (results)
5. Explain **Explainability** (transparency)

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| Install dependencies | 30 seconds |
| Run quick demo | 2-3 minutes |
| Launch web app | 5 seconds |
| **Total to demo-ready** | **~3 minutes** |
| Run full pipeline | 10-15 minutes |

---

## 🎯 Next Steps

After running the demo:

1. **Explore the app**: Try different users and methods
2. **Read PROJECT_SUMMARY.md**: Detailed findings and analysis
3. **Check evaluation_results.csv**: Raw performance metrics
4. **Review code**: Modular, well-commented implementation
5. **Git workflow**: Follow COMMITS.md for version control

---

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] Quick demo completed successfully
- [ ] Web app launches at http://localhost:8501
- [ ] Can select users and get recommendations
- [ ] Model comparison charts display correctly
- [ ] Evaluation results show Hybrid(α=0.3) as best

**All checked? You're ready to present! 🎉**

---

## 🆘 Need Help?

1. Check **README.md** for detailed documentation
2. Review **PROJECT_SUMMARY.md** for methodology
3. See **troubleshooting** section above
4. Verify all data files exist in `data/` directory

---

**Estimated time to fully functional demo: 3 minutes**

Ready? Start with:
```bash
pip install -r requirements.txt && python src/quick_demo.py && streamlit run app/app.py
```
