# Customer Sentiment Analysis

## E-Commerce Customer Sentiment Analysis (VADER + LDA)

This repository contains the Python-based natural language processing (NLP) pipeline for analyzing e-commerce customer reviews. The project uses a hybrid approach, combining rule-based sentiment scoring (VADER) with unsupervised machine learning (Latent Dirichlet Allocation) to identify thematic drivers of customer satisfaction and quantify the limitations of basic sentiment models.


## Key Features

- Automated Polarity Scoring: Generates positive, neutral, negative, and normalized compound scores for unstructured text using NLTK's SentimentIntensityAnalyzer.

- Unsupervised Topic Discovery: Implements CountVectorizer and LatentDirichletAllocation (LDA) to extract 5 distinct latent themes from the dataset, moving beyond binary sentiment.

- Custom Data Filtering: Strips standard English stop words alongside domain-specific HTML artifacts (e.g., 'br', 'href', 'quot') often scraped from e-commerce databases.

- Discrepancy Analytics: Mathematically calculates false-positive and false-negative misclassification rates by comparing the AI-generated compound scores directly against user-assigned ground-truth star ratings.  


## Tech Stack

Language: Python

Data Manipulation: pandas, numpy

NLP & Machine Learning: nltk (VADER), scikit-learn

Data Visualization: matplotlib, seaborn


## Outputs generated: 

- Vader_sentiment_high_res.png: Bar charts displaying the distribution of positive, neutral, and negative sentiments across the 1-5 star rating scale.

- lda_topics_high_res.png: A count plot showing the distribution of the 5 extracted LDA topics across the dataset.
review_distribution_high_res.png: A baseline distribution of the original dataset's star ratings.

- Console Output: The script prints the top 10 keywords for each LDA theme, a comparative average sentiment table, and the final false-positive/negative percentage rates directly to the terminal.


##
