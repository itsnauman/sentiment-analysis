# Sentiment Analysis :rocket:

I have used the Support Vector Machine for automatic classification of the subjectivity (positive / negative) of sentences.
The purpose of this projecy was to get a better understanding of how various machine learning classifiers work and perform under various conditions, i.e. do a comparative study about Sentiment Analytics.
I used a variety of models namely Naive Bayes, Logistic Regression and SVM in this project and I chose SVM at the end due to low classifiation error. I also used bi-grams to generate more complex features Furthermore, I used the GridSearch algorithm to find an optimum set of hyperparameters for the SVM.

### Usage
- Running `vocab.py` generates `vocab.txt`, which is a file with the top 1500 words from the corpus
- Generate a vocab list: `python3 vocab.py name-of-file.txt`
- The classifier can be run by `python3 sentiment_analysis.py`
