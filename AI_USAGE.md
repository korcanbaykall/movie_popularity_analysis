# AI Usage Disclosure

This document describes my use of AI tools (Claude and ChatGPT) during this project, as required by the DSA 210 course guidelines.

The AI was used in two specific areas: background research and the data collection approach. All analysis, interpretation of results, and writing of the final findings were done by me.

## 1. Research

I used Claude and ChatGPT to help me understand some methodological concepts before starting the analysis. Specifically:

- I asked about the difference between parametric and non-parametric hypothesis tests, and when each is appropriate. Based on that, I chose non-parametric tests (Mann-Whitney U, Kruskal-Wallis, Spearman) because the TMDb popularity variable is heavily right-skewed.
- I asked for a short explanation of Spearman rank correlation vs Pearson correlation, to confirm why Spearman is more suitable for skewed data.
- I asked about common ways to visualize heavily skewed distributions, which led me to add the log-scale histogram for popularity.

Example prompts I used:
- "When should I use Kruskal-Wallis instead of one-way ANOVA?"
- "Why would Spearman correlation be more appropriate than Pearson for skewed data?"
- "How do I visualize a right-skewed distribution so it is readable?"

## 2. Data Collection

I used Claude and ChatGPT to help me figure out the best way to pull data from the TMDb API and merge it with the IMDb dataset. Specifically:

- I asked how the TMDb `/find` endpoint works with an IMDb ID as an external source, which is the method I used to match each IMDb movie to its TMDb record.
- I asked about rate limiting on the TMDb API and how to avoid being throttled when fetching ~2000 movies in a loop. Based on the suggestion, I added a small `time.sleep(0.25)` delay between requests.
- I asked which fields from the TMDb `/movie/{id}` endpoint would be most useful for my project, and I decided to keep: `popularity`, `original_language`, `release_date`, `vote_average`, `vote_count`, and `genres`.

Example prompts I used:
- "How do I look up a TMDb movie using an IMDb ID?"
- "What's a safe request rate for the TMDb API when I'm fetching 2000 records?"
- "Which fields from the TMDb movie endpoint are worth keeping for a popularity analysis?"

## 3. Machine Learning Section (§9 of the notebook)

For Milestone 4 (5 May — apply ML on the dataset), I used Claude to help me structure the ML section of the notebook. Specifically:

- I asked how to frame predicting a heavily right-skewed target (TMDb popularity), and was guided to model `log1p(popularity)` for regression and to also frame a binary "top 25% popular vs not" version for classification.
- I asked whether including `numVotes` as a feature was a form of data leakage. The Spearman correlation between `numVotes` and TMDb popularity is 0.62, so it is essentially a popularity proxy. I decided to evaluate two feature sets side by side — a "full" set including votes/rating and a "metadata-only" set without them — so that the metadata-only result corresponds to the actual question my proposal asks (whether categorical features alone predict popularity).
- I asked which scikit-learn models are reasonable for a small (~968 row) tabular task and got a baseline / linear / Random Forest / Gradient Boosting comparison plus 5-fold cross-validation.
- I asked how to handle class imbalance in the classification task (~25% popular). Based on the suggestion I added `class_weight='balanced'` to Logistic Regression and Random Forest; recall on the popular class went from ~0.48 to ~0.65 on the full feature set with only a small AUC change.
- I asked for a small, reasonable `GridSearchCV` grid for the Gradient Boosting regressor (`n_estimators`, `max_depth`, `learning_rate`) and used 5-fold inner CV with the held-out test set untouched for the final number.

Example prompts I used:
- "Why log1p the target if popularity is right-skewed?"
- "Is it a problem to use numVotes as a feature when predicting TMDb popularity?"
- "How do I handle the imbalanced popular class in scikit-learn?"
- "Suggest a small GridSearchCV grid for GradientBoostingRegressor on a ~1000-row dataset."

## Scope of AI Use

The AI was used for methodological guidance, code structure suggestions, and explanations of when each technique is appropriate. The interpretation of results, the decision to evaluate two feature sets side by side, the decision to keep the proposal-aligned metadata-only set as the headline result, and the final conclusions in the summary section reflect my own work. All code was reviewed and run by me.
