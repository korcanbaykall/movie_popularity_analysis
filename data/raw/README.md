# Raw Data Sources

## IMDb Non-Commercial Datasets

The raw IMDb datasets are too large for GitHub (title.basics.tsv.gz ~210 MB).
Download them from the official source:

- **Source:** https://datasets.imdbws.com/
- **Files needed:**
  - `title.basics.tsv.gz` — movie metadata (titles, genres, year, runtime)
  - `title.ratings.tsv.gz` — user ratings and vote counts (included in this repo)

Place both `.tsv.gz` files in this directory before running the notebook.

## TMDb API

TMDb data is fetched via the TMDb API v3 during notebook execution.
You need a free API Read Access Token from https://www.themoviedb.org/settings/api
