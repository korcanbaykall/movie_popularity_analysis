# Final Report
## Analyzing and Predicting Movie Popularity from Genres and Categorical Features Using IMDb and TMDb Data

**Course:** DSA 210 — Introduction to Data Science, Spring 2025–2026
**Student:** Korcan Baykal
**Repository:** https://github.com/korcanbaykall/movie_popularity_analysis
**Notebook:** [`notebooks/DSA210_Movie_Popularity_Analysis.ipynb`](notebooks/DSA210_Movie_Popularity_Analysis.ipynb)

---

## 1. Motivation

Movies are one of the most widely consumed forms of media, and their commercial and cultural success is shaped by a long list of factors: genre, release timing, language, audience engagement, and many production-side variables that are not easy to observe from the outside. As a data science student and an active consumer of films, I wanted to look at this from the metadata side and ask a simple question: **how far can the categorical features of a movie alone take us in explaining and predicting its popularity?**

This project pursues that question in three stages. First, it characterizes how popularity is distributed across genres, languages, and time. Second, it tests whether the apparent differences between genres are statistically significant or could plausibly arise from noise. Third, it builds predictive models on two intentionally different feature sets — one that includes engagement signals (IMDb vote count and average rating), and one that uses categorical metadata only — so that the contribution of each kind of feature can be read off directly.

Beyond the substantive question, the project is a practice run through the full data science pipeline: collecting and merging two public data sources, cleaning a large raw dataset, performing exploratory analysis, running non-parametric hypothesis tests appropriate for a heavily skewed target, and training and tuning regression and classification models with proper train/test discipline and cross-validation.

---

## 2. Data Source

The analysis combines two public data sources, which together satisfy the course requirement of enriching publicly available data with another source.

**IMDb Non-Commercial Datasets** ([datasets.imdbws.com](https://datasets.imdbws.com/)) — bulk TSV files refreshed daily:
- `title.basics.tsv.gz` (~210 MB): `tconst`, primary title, title type, isAdult, start year, runtime, genres.
- `title.ratings.tsv.gz`: `tconst`, average rating, number of votes.

**TMDb API** ([themoviedb.org](https://www.themoviedb.org/)) — per-movie metadata fetched programmatically:
- `popularity` (TMDb's internally computed score), `original_language`, `release_date`, `vote_count`, `vote_average`, and a TMDb genre list.

**Collection and merging pipeline** (Notebook §1–§5):

1. Both IMDb TSVs were downloaded and read with pandas.
2. `title.basics` was filtered to `titleType == "movie"` and `isAdult == 0` to keep general-audience films, then merged with `title.ratings` on `tconst`.
3. Numeric columns (`startYear`, `runtimeMinutes`, `averageRating`, `numVotes`) were coerced to numeric. Rows missing `primaryTitle`, `startYear`, or `genres` were dropped, and duplicates removed. **Final IMDb-only cleaned dataset: 553,802 movies.**
4. A random sample of **2,000 movies** was drawn and enriched by calling the TMDb `/find/{imdb_id}` endpoint (with `external_source=imdb_id`) to resolve the TMDb id, then `/movie/{tmdb_id}` to pull the movie record. The fetched results are cached as a CSV (`data/processed/movies_imdb_tmdb_2000.csv`) so the rest of the notebook is reproducible without a TMDb API key.
5. The IMDb-TMDb merged dataset yields **1,251 movies with non-null popularity** and **968 movies with all features non-null** (the modeling set in §9).

The raw `title.basics.tsv.gz` is gitignored due to size; the README documents the one-line `curl`/`wget` command to re-download it.

---

## 3. Data Analysis

The analysis is organized into four stages, each occupying a numbered section of the notebook.

### 3.1 Cleaning and feature engineering (Notebook §1–§5)

Beyond the type coercion and null/duplicate removal described above, the merged dataset was extended with derived features used later in the modeling step: `log_numVotes = log1p(numVotes)` to compress the heavy tail, `decade` bins from `startYear`, multi-hot genre indicators from the comma-separated IMDb `genres` column, and one-hot indicators for the top eight `original_language` values (with the rest grouped as "other").

### 3.2 Exploratory data analysis (Notebook §4, §6, §7)

EDA covered marginal distributions, pairwise relationships, and group-level comparisons:

| Figure | What it shows |
|---|---|
| `imdb_rating_distribution.png` | IMDb ratings are roughly normally distributed around 6.0–7.0. |
| `top10_genres.png` | Drama, Documentary, and Comedy are the most common genres. |
| `popularity_log_hist.png` | TMDb popularity is heavily right-skewed; a log transform makes the tail readable. |
| `genre_popularity_boxplot.png` | Median popularity differs by genre; Action and Adventure sit highest. |
| `rating_vs_popularity.png` | No strong linear pattern between IMDb rating and TMDb popularity. |
| `top10_languages.png` | English dominates; French, Japanese, and Spanish follow. |

The right-skew on the target is the most important EDA observation: it motivated both the choice of non-parametric tests in §8 and the `log1p` transformation of the regression target in §9.

### 3.3 Hypothesis tests (Notebook §8)

Three tests on the **1,045-movie subset** with non-null popularity, genres, and IMDb rating, at significance level **α = 0.05**. All tests are non-parametric because the popularity distribution violates normality assumptions:

| Test | Method | Hypothesis | Result | p-value |
|---|---|---|---|---|
| A: popularity across top 5 genres (project's main question) | Kruskal-Wallis | H₀: all genres equally popular | **Reject H₀** | < 0.001 |
| B: Action vs non-Action popularity | Mann-Whitney U | H₀: distributions identical | **Reject H₀** (median 0.635 vs 0.364) | < 0.001 |
| C: IMDb rating ↔ TMDb popularity | Spearman correlation | H₀: ρ = 0 | **Reject H₀** (ρ = −0.14, weak negative) | < 0.001 |

Test A is the formal version of the project's main question and provides the statistical justification for moving to ML. Test C is interesting in its own right: even though the relationship is statistically significant due to sample size, the magnitude is small and *negative*, meaning IMDb rating and TMDb popularity capture different things — TMDb popularity tracks present-day audience attention more than critical quality.

### 3.4 Machine learning (Notebook §9)

The 968-row modeling set was split **80/20** into train and test, stratified on the classification target, with `random_state=42`. **5-fold cross-validation** on the training set was used to check stability before reporting test-set numbers.

**Two feature sets are evaluated side by side** to isolate the contribution of engagement features:

- **Full feature set:** `runtimeMinutes`, `averageRating`, `log_numVotes`, `startYear`, decade, top-8 language indicators, multi-hot genre indicators.
- **Metadata-only feature set:** the same set with `averageRating` and `log_numVotes` removed. `numVotes` correlates strongly with popularity (Spearman ρ ≈ 0.62) and is essentially an engagement proxy; the metadata-only set is the version that matches the proposal's framing — *can categorical metadata alone predict popularity?*

**Regression** (target: `log1p(popularity)`) — four models per feature set: median baseline, Linear Regression, Random Forest, Gradient Boosting.

**Classification** (target: in top 25% of popularity, ~25/75 imbalance) — Logistic Regression and Random Forest with `class_weight='balanced'`; Gradient Boosting (scikit-learn's GB does not accept `class_weight`); plus a "predict not popular" baseline.

**Hyperparameter tuning** — `GridSearchCV` (5-fold) over the **metadata-only** Gradient Boosting regressor, with grid `n_estimators ∈ {200, 400}`, `max_depth ∈ {2, 3, 4}`, `learning_rate ∈ {0.03, 0.05, 0.1}` (18 configurations). The best configuration is `n_estimators=200, max_depth=2, learning_rate=0.05`, reported as the final tuned regressor on the held-out test set.

---

## 4. Findings

### 4.1 Statistical findings

- **Popularity differs significantly across genres.** The main hypothesis of the project is confirmed by Kruskal-Wallis (p < 0.001), and the boxplot in `genre_popularity_boxplot.png` shows the practical effect: Action and Adventure sit visibly above Documentary and Drama.
- **Action is a popularity-relevant genre.** The Action / non-Action median gap (0.635 vs 0.364) is large in addition to being statistically significant.
- **IMDb rating is not a substitute for TMDb popularity.** The two have a weak negative Spearman correlation (ρ = −0.14). High-rated films are not systematically the most popular ones in TMDb's sense, which is a useful corrective: critical reception and audience attention measure different signals.

### 4.2 Predictive modeling findings

**Regression — `log1p(popularity)`:**

| Feature set | Best model | Test R² | Test MAE | Test RMSE | 5-fold CV R² |
|---|---|---|---|---|---|
| Full | Gradient Boosting | **0.594** | 0.146 | 0.208 | 0.490 |
| Metadata-only (default) | Random Forest | **0.306** | 0.197 | 0.272 | 0.177 |
| Metadata-only (tuned GB) | Gradient Boosting | **0.319** | 0.196 | 0.269 | 0.177 |
| Either | Median baseline | −0.066 | 0.229 | 0.337 | — |

**Classification — popular (top 25%) vs not:**

| Feature set | Best model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Full | Random Forest (balanced) | 0.856 | 0.738 | 0.646 | 0.689 | **0.878** |
| Full | Gradient Boosting | 0.845 | 0.750 | 0.562 | 0.643 | 0.875 |
| Metadata-only | Random Forest (balanced) | 0.789 | 0.581 | 0.521 | 0.549 | **0.801** |

Three findings deserve emphasis:

1. **Metadata alone is informative but not sufficient.** Without `numVotes` and `averageRating`, the regressor still reaches R² ≈ 0.32 — five-fold better than the median baseline (R² = −0.07) — and the classifier still reaches ROC-AUC ≈ 0.80. So categorical metadata genuinely carries predictive signal, which directly answers the project's main question.
2. **Engagement features close roughly half the remaining gap.** Adding `log_numVotes` and `averageRating` raises regression R² from 0.32 to 0.59 and classification AUC from 0.80 to 0.88. The 0.27-point R² and 0.08-point AUC differences are the share of predictive power that engagement features carry as a popularity proxy.
3. **Class balancing matters for the rare-class recall.** Switching to `class_weight='balanced'` raises Random Forest's recall on the popular class from ~0.48 (unweighted) to 0.65 on the full feature set, at the cost of about one point of ROC-AUC — a worthwhile trade for a 25/75 imbalance.

**Top features (Gradient Boosting feature importance):**
- **Full set:** `log_numVotes` and `averageRating` dominate, followed by `runtimeMinutes` and a few genre flags.
- **Metadata-only set:** `runtimeMinutes`, `startYear`, several genre flags (Documentary, Thriller, Action, Sci-Fi), and the Japanese- and English-language indicators.

Visual evidence is in `ml_pred_vs_actual_full.png`, `ml_pred_vs_actual_meta.png`, `ml_feature_importance.png`, `ml_roc_curves.png`, and `ml_confusion_matrix.png`.

---

## 5. Limitations and Future Work

### 5.1 Limitations

- **Sample size.** 968 fully observed rows is small for a model with this many categorical features; the gap between cross-validated and test-set R² on the metadata-only regressor (0.18 vs 0.32) suggests modest variance in the estimates.
- **Temporal instability of the target.** TMDb popularity is recomputed daily, so a static snapshot taken in spring 2026 will not generalize cleanly to other points in time. The model captures the structure of popularity *at the moment of collection*, not a stable underlying quantity.
- **Limited feature breadth.** The metadata covers genre, language, decade, and runtime, but not budget, cast, director, studio, or marketing spend — exactly the features one would expect to drive popularity in practice. The 0.32 R² ceiling on metadata-only is probably as much a missing-feature issue as a fundamental information limit.
- **Class imbalance.** Even with `class_weight='balanced'`, the popular class is under-recalled (0.65 on the best classifier). Applications that want to find popular films at high recall would need either threshold tuning or cost-sensitive learning beyond what the default balancing provides.
- **Sampling bias.** The 2,000-movie TMDb sample is drawn at random from the cleaned IMDb pool, but TMDb coverage is itself biased toward more widely known films, so titles with very low engagement may be under-represented in the merged set.

### 5.2 Future work

- **Widen the TMDb sample** to 10,000+ movies; the API rate limits make this an overnight job, not a redesign.
- **Add credits-based features** (top-billed cast, director, production company) via the TMDb `/movie/{id}/credits` endpoint. These are the most plausible candidates to close the gap between the metadata-only and full-feature models without re-using `numVotes` as a proxy.
- **Model popularity over time.** Pull TMDb popularity weekly for a fixed sample for a few months, and frame the task as predicting future popularity from past metadata plus past popularity — a more meaningful question than predicting the snapshot.
- **Threshold tuning and cost-sensitive learning** for the classification task, to push popular-class recall higher than the 0.65 ceiling reached here.
- **Text and image features.** Title, overview, and poster image embeddings from TMDb would let a future iteration test whether content-level features (not just categorical metadata) add signal on top of the structural features used here.

---

## Appendix — Reproducibility

All code is in [`notebooks/DSA210_Movie_Popularity_Analysis.ipynb`](notebooks/DSA210_Movie_Popularity_Analysis.ipynb), which executes end-to-end. Dependencies are listed in `requirements.txt`. Re-running the analysis requires re-downloading `data/raw/title.basics.tsv.gz` (~210 MB) — the README has the one-line command. The 2,000-row TMDb-enriched CSV is committed, so reproduction does not require a TMDb API key. Random seeds are fixed (`random_state=42`).

AI assistance during the project is disclosed in [`AI_USAGE.md`](AI_USAGE.md).
