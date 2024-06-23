
# AFame Tech SMS Detection

This repository contains files related to SMS spam detection using machine learning techniques. Below is an overview of the files included:

## Files Included:

### sms_spam_classifier.pkl
- **Description:** This file contains a trained machine learning model for classifying SMS messages as either spam or legitimate ('ham').
- **Usage:** The model can be loaded using joblib in Python for predicting the classification of new SMS messages.

### sms_spam_classifier_naive_bayes.pkl
- **Description:** This file specifically contains a Naive Bayes model trained for SMS spam detection.
- **Usage:** It can be loaded using joblib in Python for classifying new SMS messages based on the Naive Bayes algorithm.

### spam.csv
- **Description:** This CSV file contains a dataset of SMS messages tagged as spam or ham.
- **Usage:** Used for training and testing the SMS spam detection models. It includes two columns: v1 for the label ('ham' or 'spam') and v2 for the SMS message text.

### tfidf_vectorizer.pkl
- **Description:** This file stores a TF-IDF vectorizer that has been fitted on the SMS message data.
- **Usage:** The vectorizer is used to transform text data into numerical features based on TF-IDF (Term Frequency-Inverse Document Frequency) weighting, which is essential for input to the machine learning models.

## Usage:

- **Training the Model:** Use spam.csv to train the machine learning models (sms_spam_classifier.pkl and sms_spam_classifier_naive_bayes.pkl) using Python scripts or Jupyter Notebooks.
- **Classification:** After training, load the appropriate model (sms_spam_classifier.pkl or sms_spam_classifier_naive_bayes.pkl) and TF-IDF vectorizer (tfidf_vectorizer.pkl) using joblib. Use these to classify new SMS messages as 'spam' or 'ham'.

## Dependencies:

- Python 3.x
- Pandas
- NLTK
- Scikit-learn (sklearn)
- Joblib

## Notes:

- Ensure spam.csv is accessible in the working directory or update the file path accordingly in your scripts.
- The models (sms_spam_classifier.pkl and sms_spam_classifier_naive_bayes.pkl) are trained on data from spam.csv. Adjustments may be necessary for improved accuracy on different datasets or applications.

---
