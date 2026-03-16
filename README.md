# Movie Popularity Analysis

## Project Title
**Analyzing and Predicting Movie Popularity from Genres and Categorical Features Using IMDb and TMDb Data**

## Overview
This project aims to analyze and predict movie popularity by combining data from **IMDb** and **TMDb**. The main goal is to investigate how genres and other categorical movie features are associated with popularity, and whether these features can be used to build predictive models.

This project is developed as part of the **DSA 210 Introduction to Data Science** term project. The course requires students to work individually, use Python, maintain a GitHub repository throughout the project, enrich publicly available data with another data source, and include both a `requirements.txt` file and a `README.md` with reproduction instructions. 

## Motivation
Movie success and popularity are influenced by many factors, including genre, release period, audience engagement, and metadata such as language or production-related attributes. By combining data from IMDb and TMDb, this project seeks to better understand which categorical features are most related to popularity and to explore whether movie popularity can be predicted from these variables.

## Data Sources
This project uses two public sources:

- **IMDb**: movie-related information such as titles, genres, ratings, and vote counts
- **TMDb**: additional metadata such as popularity, release date, original language, and other movie descriptors

Using both sources allows the dataset to be enriched, which matches the project expectation for publicly available datasets. 

## Main Objectives
- Collect movie data from IMDb and TMDb
- Merge the two datasets into a single analysis-ready dataset
- Analyze the relationship between genres, categorical features, and movie popularity
- Perform exploratory data analysis (EDA) and hypothesis testing
- Build machine learning models to predict movie popularity
- Interpret results and discuss limitations

## Planned Workflow
1. **Data Collection**
   - Obtain movie-related data from IMDb
   - Gather additional metadata from TMDb
   - Match and merge records across the two sources

2. **Data Preprocessing**
   - Clean missing or inconsistent values
   - Select relevant features
   - Encode categorical variables where necessary

3. **Exploratory Data Analysis**
   - Examine the distribution of popularity
   - Compare popularity across genres and other categories
   - Visualize important trends and patterns

4. **Hypothesis Testing**
   - Test whether certain genres or categories are significantly associated with higher popularity

5. **Modeling**
   - Train machine learning models for popularity prediction
   - Evaluate model performance using appropriate metrics

6. **Reporting**
   - Summarize findings
   - Discuss limitations and future improvements

## Repository Structure
```text
movie-popularity-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
└── figures/
