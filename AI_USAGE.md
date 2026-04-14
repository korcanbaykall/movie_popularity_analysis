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

## Scope of AI Use

The AI was not used to write the final analysis, decide which hypotheses to test, interpret the results, or produce the conclusions in the report. Those parts reflect my own work. The code for EDA, hypothesis tests and the final analysis was written and run by me.
