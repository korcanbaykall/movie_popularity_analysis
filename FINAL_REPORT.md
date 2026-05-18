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

**Collection and merging pipeline** (Notebook sections 1–5):

1. Both IMDb TSVs were downloaded and read with pandas.
2. `title.basics` was filtered to `titleType == "movie"` and `isAdult == 0` to keep general-audience films, then merged with `title.ratings` on `tconst`.
3. Numeric columns (`startYear`, `runtimeMinutes`, `averageRating`, `numVotes`) were coerced to numeric. Rows missing `primaryTitle`, `startYear`, or `genres` were dropped, and duplicates removed. **Final IMDb-only cleaned dataset: 553,802 movies.**
4. A random sample of **2,000 movies** was drawn and enriched by calling the TMDb `/find/{imdb_id}` endpoint (with `external_source=imdb_id`) to resolve the TMDb id, then `/movie/{tmdb_id}` to pull the movie record. The fetched results are cached as a CSV (`data/processed/movies_imdb_tmdb_2000.csv`) so the rest of the notebook is reproducible without a TMDb API key.
5. The IMDb-TMDb merged dataset yields **1,251 movies with non-null popularity** and **968 movies with all features non-null** (the modeling set in section 9).

The raw `title.basics.tsv.gz` is gitignored due to size; the README documents the one-line `curl`/`wget` command to re-download it.

---

## 3. Data Analysis

The analysis is organized into four stages, each occupying a numbered section of the notebook.

### 3.1 Cleaning and feature engineering (Notebook sections 1–5)

Beyond the type coercion and null/duplicate removal described above, the merged dataset was extended with derived features used later in the modeling step: `log_numVotes = log1p(numVotes)` to compress the heavy tail, `decade` bins from `startYear`, multi-hot genre indicators from the comma-separated IMDb `genres` column, and one-hot indicators for the top eight `original_language` values (with the rest grouped as "other").

### 3.2 Exploratory data analysis (Notebook sections 4, 6, 7)

EDA covered marginal distributions, pairwise relationships, and group-level comparisons. All figures referenced below are saved under `figures/` and reproduced inline in the notebook.

#### 3.2.1 Marginal distributions of the headline variables

**`imdb_rating_distribution.png` — Distribution of IMDb average rating.** Histogram of the `averageRating` column over the full 553,802-row cleaned IMDb dataset. The distribution is roughly bell-shaped with a peak around 6.5 and a mild left skew. The bulk of mass lies between 5 and 8, ratings below 3 or above 9 are rare, and there is no extreme tail. This is the most well-behaved variable in the project — it can be fed directly into linear models without transformation.

**`imdb_votes_distribution.png` — Distribution of IMDb number of votes.** Histogram of the raw `numVotes` column. In stark contrast to the rating, this is the most heavy-tailed variable in the dataset: more than 99% of films have fewer than ~100,000 votes, so the histogram is dominated by a single tall bar near zero, while a thin tail extends past 3 million. Values span roughly seven orders of magnitude on the raw scale. This figure is the visual justification for the derived feature `log_numVotes = log1p(numVotes)` used in section 9 — without the log transform, linear models would be dominated entirely by a handful of blockbuster titles.

**`tmdb_popularity_distribution.png` — Distribution of raw TMDb popularity.** Histogram of TMDb's `popularity` field over the 1,251 merged IMDb–TMDb rows with a non-null value. The shape is strongly right-skewed: ~815 films sit in the lowest bin (0–0.5), ~280 in the next, and the tail thins quickly out to about 14 with only a handful of points beyond 4. Working with this variable on its raw scale would let a few outliers dominate any squared-error regression.

**`popularity_log_hist.png` — Distribution of TMDb popularity (log scale).** The same target after a `log1p` transformation. The distribution becomes much more readable: a clear mode near 0.1–0.3, a smooth taper through ~1.0, and a thin tail reaching ~2.5. This is the scale the section 9 regression models are trained against, both for numerical stability and because residuals are far closer to symmetric here than on the raw scale.

#### 3.2.2 Categorical-feature counts

**`top10_genres.png` — Top 10 IMDb genres by frequency.** Bar chart of the 10 most common genre labels across the 553,802-row IMDb cleaned dataset (genres are multi-hot, so totals overlap). Drama is by far the most common (~235k titles), followed by Documentary (~135k) and Comedy (~105k); Action, Romance, Crime, Thriller, Horror, Adventure, and Family each have 17k–50k titles. The takeaway is purely about availability: being a common genre is not the same as being a popular one. Drama is the most-produced genre, but it sits near the middle of the popularity boxplot below.

**`top10_languages.png` — Top 10 TMDb original languages.** Bar chart of `original_language` counts on the 1,251-row TMDb-enriched subset. English (`en`, ~525) overwhelmingly dominates — more than five times any other single language. Spanish (`es`, ~90), French (`fr`, ~50), Italian (`it`, ~48), Japanese (`ja`, ~44), German (`de`, ~43), Russian (`ru`, ~40), Chinese (`zh`, ~37), Turkish (`tr`, ~26), and Portuguese (`pt`, ~24) round out the top 10. This imbalance is what drove the section 9 design choice to one-hot-encode only the top 8 languages and group the rest into a single `other` indicator — otherwise dozens of sparse language columns would inflate the feature space without carrying meaningful signal.

#### 3.2.3 Pairwise relationships

**`rating_vs_votes.png` — IMDb average rating versus number of votes.** Scatter plot of a random 5,000-row sample from the cleaned IMDb data. The pattern is a clear triangle/cone shape: films at the rating extremes (below 3 or above 9) almost always have very few votes, while films with high vote counts (50,000+) cluster in the 6–8 rating band. Unanimous very-low or very-high ratings are essentially a small-sample artefact. This figure also argues against collapsing rating and votes into a single "engagement" feature — they carry different information, and section 9 keeps both.

**`rating_vs_popularity.png` — IMDb rating versus TMDb popularity.** Scatter of the two key cross-source variables on the 1,251-row merged dataset. There is no strong linear relationship — the scatter is a wide cloud — and visually, very high popularity values occur across a range of ratings rather than concentrating at the top of the rating axis. Several of the most-popular outliers sit in the 6–8 rating range rather than at 9+. This figure is the visual companion of hypothesis Test C in section 8, which finds a small but statistically significant *negative* Spearman correlation of ρ = −0.14: critical reception (IMDb rating) and audience attention (TMDb popularity) measure substantively different things.

**`votes_vs_popularity.png` — IMDb number of votes versus TMDb popularity.** Scatter of `numVotes` against TMDb `popularity`. In sharp contrast to the previous plot, there is a clear positive trend: films with hundreds of thousands of IMDb votes are almost always above-median in TMDb popularity, and the few highest-popularity points (>6) all sit at very high vote counts. The Spearman correlation here is ρ ≈ 0.62. This is precisely why the section 9 modeling step evaluates a separate *metadata-only* feature set — keeping `numVotes` in a model that predicts popularity would amount to using one form of audience engagement to predict another.

#### 3.2.4 Group-level comparison

**`genre_popularity_boxplot.png` — TMDb popularity by primary genre.** Boxplots (outliers hidden for readability) of TMDb popularity for the eight most common primary genres on the 1,045-row subset with non-null popularity, genres, and rating. The medians are clearly not equal: Action sits highest (~0.60), followed by Crime (~0.53), Adventure (~0.50), Comedy (~0.41), Drama (~0.37), Horror (~0.30), Biography (~0.18), and Documentary (~0.10). Documentary is roughly six times less popular at the median than Action, and its interquartile range is also far tighter. The visible gap between the boxes is the descriptive justification for the formal Kruskal-Wallis test in section 8, which rejects the null of equal medians at p < 0.001 and supports the project's central hypothesis that genre carries real information about popularity.

---

The right-skew on the popularity target — visible in both `imdb_votes_distribution.png` and `tmdb_popularity_distribution.png` — is the single most consequential EDA observation. It motivated both the choice of non-parametric tests in section 8 (which do not assume normality) and the `log1p` transformation of the regression target in section 9.

### 3.3 Hypothesis tests (Notebook section 8)

Three tests on the **1,045-movie subset** with non-null popularity, genres, and IMDb rating, at significance level **α = 0.05**. All tests are non-parametric because the popularity distribution violates normality assumptions:

| Test | Method | Hypothesis | Result | p-value |
|---|---|---|---|---|
| A: popularity across top 5 genres (project's main question) | Kruskal-Wallis | H₀: all genres equally popular | **Reject H₀** | < 0.001 |
| B: Action vs non-Action popularity | Mann-Whitney U | H₀: distributions identical | **Reject H₀** (median 0.635 vs 0.364) | < 0.001 |
| C: IMDb rating ↔ TMDb popularity | Spearman correlation | H₀: ρ = 0 | **Reject H₀** (ρ = −0.14, weak negative) | < 0.001 |

Test A is the formal version of the project's main question and provides the statistical justification for moving to ML. Test C is interesting in its own right: even though the relationship is statistically significant due to sample size, the magnitude is small and *negative*, meaning IMDb rating and TMDb popularity capture different things — TMDb popularity tracks present-day audience attention more than critical quality.

### 3.4 Machine learning (Notebook section 9)

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
- **Full set:** `log_numVotes` alone accounts for over 60% of total importance, dwarfing every other variable; `lang_top_ja`, `startYear`, `averageRating`, and `runtimeMinutes` follow with much smaller bars.
- **Metadata-only set:** no single feature dominates — `runtimeMinutes` (~0.22) and `startYear` (~0.19) lead, followed by `lang_top_ja` and `genre_Documentary` (~0.10 each), then a long tail of genre flags (Thriller, Action, Sci-Fi, Adventure, Fantasy, Animation, Horror) and the English-language indicator.

#### 4.2.1 Visual walkthrough of the modeling figures

**`ml_pred_vs_actual_full.png` — Predicted vs actual `log1p(popularity)`, full feature set.** Four side-by-side scatter panels (Baseline median, Linear Regression, Random Forest, Gradient Boosting), each plotting predicted values against actual values on the held-out test set with the red 45° dashed line marking perfect prediction. The Baseline panel collapses to a horizontal stripe at the training median (~0.32) — by construction, the median model cannot vary its prediction. Linear Regression aligns roughly along the diagonal but compresses the high end, systematically under-predicting actual values above ~1.0. Random Forest and Gradient Boosting both track the diagonal more tightly in the central mass of the data, with GB being the best overall (test R² = 0.594). All three trained models share one clear failure mode: the few very high-popularity points in the upper-right corner (actual log1p ≈ 1.8–2.2) are consistently pulled toward the centre — even the best model puts the most extreme point at only ≈ 1.0. This is the visual signature of the dataset's heavy right tail combined with a small number of training examples in that range.

**`ml_pred_vs_actual_meta.png` — Predicted vs actual, metadata-only feature set.** The same four-panel layout, but trained without `log_numVotes` and `averageRating`. The Baseline panel is identical (it does not use features). The trained-model panels show visibly more scatter around the diagonal, more compression at the high end, and a tighter cluster of predictions near the central mode. The regression R² drops from ≈ 0.59 (full) to ≈ 0.32 (metadata-only), and this figure makes the geometry of that loss concrete: the metadata-only model can still place most points roughly in the right region of the popularity axis, but it can no longer pull the high-popularity examples up the diagonal at all — its predictions for actual log1p > 1.0 are barely distinguishable from its predictions for actual log1p ≈ 0.5.

**`ml_feature_importance.png` — Top 15 feature importances (Gradient Boosting).** Two horizontal bar charts side by side, one per feature set. On the **full** side, `log_numVotes` alone occupies more than 60% of the total importance budget, with `lang_top_ja`, `startYear`, `averageRating`, and `runtimeMinutes` trailing far behind. This figure visualises numerically what the EDA already implied: when an engagement signal is available, the model essentially becomes a one-feature predictor. On the **metadata-only** side, the profile is qualitatively different — no single feature dominates. `runtimeMinutes` and `startYear` lead with ~0.22 and ~0.19, then `lang_top_ja` and `genre_Documentary` cluster around 0.10, followed by a long tail of genre flags (Thriller, Action, Sci-Fi, Adventure, Fantasy, Animation, Horror) and the English-language indicator. The metadata-only model is genuinely *combining* runtime, era, language, and genre signals rather than reading off one dominant variable. The persistent prominence of the Japanese-language indicator is interesting and most likely reflects animation/anime films, which are over-represented at the top of TMDb's popularity score in the 2026 snapshot.

**`ml_roc_curves.png` — ROC curves for the three classifiers.** Two panels (full vs metadata-only), each plotting the true-positive rate against the false-positive rate for Logistic Regression, Random Forest, and Gradient Boosting; the dashed diagonal is the no-skill baseline (AUC = 0.5). On the **full** side all three curves rise sharply toward the top-left — Random Forest leads at AUC = 0.878, with Gradient Boosting (0.875) almost overlapping it and Logistic Regression slightly below (0.855). The curves stay well above 0.8 true-positive rate even at low false-positive rates, meaning the model can find most popular films while flagging relatively few non-popular ones. On the **metadata-only** side the curves are visibly less aggressive but still well above the diagonal: Random Forest 0.801, Gradient Boosting 0.793, Logistic Regression 0.751. The vertical gap between the two panels at any fixed false-positive rate is the geometric meaning of the "engagement features add ≈ 0.08 of AUC" finding tabulated above — and that gap is widest in the low-FPR region, which is exactly where high-precision applications would operate.

**`ml_confusion_matrix.png` — Confusion matrices for the best classifier on each feature set.** Two heatmaps in shades of blue, for Random Forest with `class_weight='balanced'` (the highest-AUC model in both feature sets) on the 194-row test set. The **full-feature** matrix has 135 true negatives (Not popular → Not popular), 31 true positives (Popular → Popular), 11 false positives, and 17 false negatives. The **metadata-only** matrix degrades all four cells modestly: 128 TN, 25 TP, 18 FP, 23 FN. Two observations follow. (i) Both models are biased toward predicting "Not popular", which is expected given the 25/75 class imbalance even after balancing weights. (ii) Most of the accuracy loss when removing engagement features falls on the *positive* class: recall on Popular drops from 31/48 ≈ 65% (full) to 25/48 ≈ 52% (metadata-only), while specificity on Not popular only drops from 93% to 88%. The classifier suffers most exactly where the prediction task is hardest — identifying the popular minority from structural metadata alone.

---

Read together, these five figures tell the same coherent story: when an engagement signal is available, prediction is largely a one-feature problem (`log_numVotes`) and accuracy is high; when only structural metadata is available, the model has to combine many weaker signals (runtime, era, language, genre flags), and the predictive ceiling is meaningfully lower but still well above baseline.

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
