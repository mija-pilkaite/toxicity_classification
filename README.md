### Toxicity Classification of Wikipedia Comments

# Introduction
In the digital age, social media and online interactions are pivotal in our daily communications. However, this interaction space is often marred by toxic and unhealthy comments, which can hinder constructive dialogue. Our project focuses on the detection and quantification of toxicity in Wikipedia comments, leveraging machine-learning techniques to identify harmful interactions effectively.

# Team Members
Anja Matic
Mija Pilkaite
Emma Moncia

# Project Overview
We developed a machine-learning model to classify comments into toxic and non-toxic categories and quantify the level of toxicity. The dataset, sourced from the Wikipedia Talk Corpus on Kaggle, consists of annotated comments that have been evaluated for various toxic behaviors.

# Techniques and Technologies
Feature Representation: Bag of Words, TF-IDF, SBert
Models Used: Logistic Regression, SGD Regressor, Decision Tree Regressor
Evaluation Metrics: Mean Squared Error (MSE), Accuracy Score

# Highlights
Data Analysis: Comprehensive analysis of the dataset to understand common toxicity types.
Model Development: Application of multiple sentence transformation techniques alongside prediction models to determine the most effective approach.
Feature Exploration: Use of various feature representations to encapsulate the textual data's essence fully.
Performance Comparison: Evaluation of different models on various text embeddings to optimize performance.

# Getting Started
Prerequisites
Python 3.11
Jupyter Notebook
Required Libraries: `pip install numpy pandas scikit-learn sentence-transformers`

# Installation
Clone the repository to your local machine:
```
git clone https://github.com/yourusername/toxicity-classification.git
cd toxicity-classification
```
# Appendices
The repository includes detailed Jupyter notebooks with code and analysis, along with references to external sources for natural language processing techniques.

# References
- **NLTK Stemming**
  Detailed documentation on the stemming capabilities provided by NLTK, useful for text preprocessing in NLP applications. [View Documentation](https://www.nltk.org/api/nltk.stem.snowball.html)

- **Imbalanced-Learn**
  Provides techniques for under-sampling a dataset, useful for handling class imbalance in machine learning datasets. [Learn More](https://imbalanced-learn.org/stable/under_sampling.html)

- **Bag of Words Tutorial**
  An introductory guide to the Bag of Words technique, which explains the basic concept and implementation in data processing. [Read Tutorial](https://www.mygreatlearning.com/blog/bag-of-words/)

- **Scikit-Learn CountVectorizer**
  Documentation for `CountVectorizer`, which is a tool that converts a collection of text documents to a matrix of token counts. [Scikit-Learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

- **TF-IDF Vectorizer**
  Explanation of the TF-IDF technique and its implementation in Scikit-Learn for feature extraction from text. [Scikit-Learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

- **Sentence Embeddings**
  Overview of generating sentence embeddings using the Sentence Transformers library, ideal for advanced NLP tasks. [Sentence Transformers](https://www.sbert.net/)

- **Pinecone Sentence Embeddings**
  Discusses how to use Pinecone to manage data with sentence embeddings in applications. [Pinecone Info](https://www.pinecone.io/learn/sentence-embeddings/)

# Acknowledgments
Kaggle for providing the dataset [Kaggle dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)

Anja Matic and Emma Moncia for their dedicated contributions to the project.
