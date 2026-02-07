# cyberbullying_detection_logistic_regression.py
This Python project implements a complete Cyberbullying Detection System using Natural Language Processing (NLP) and Logistic Regression. It includes data preprocessing, text cleaning, TF-IDF feature extraction, model training, evaluation, and visualizations such as word clouds, confusion matrix, and ROC curves.
# Cyberbullying Detection Using Logistic Regression

##  Project Overview
This project focuses on detecting different types of **cyberbullying** from social media text using **Natural Language Processing (NLP)** and **Machine Learning**.  
A **Logistic Regression** model is trained on preprocessed tweet data to classify cyberbullying categories effectively.

---

##  Objectives
- Clean and preprocess raw tweet text
- Extract meaningful features using **TF-IDF**
- Train a **Logistic Regression** classifier
- Evaluate model performance using multiple metrics
- Visualize insights with word clouds, PCA, confusion matrix, and ROC curves

---

##  Technologies Used
- Python
- Pandas, NumPy
- NLTK (Text Preprocessing)
- Scikit-learn
- Matplotlib & Seaborn
- WordCloud
- Joblib

---

##  Dataset
- Dataset consists of tweets labeled with different **cyberbullying types**
- Duplicate, null, and irrelevant records are removed
- One class (`other_cyberbullying`) is excluded for better classification

---

##  Project Workflow
1. **Data Loading**
2. **Text Preprocessing**
   - Lowercasing
   - Removing URLs, emojis, punctuation
   - Stopword removal
   - Stemming
3. **Visualization**
   - Word clouds (before & after preprocessing)
4. **Feature Extraction**
   - TF-IDF (unigrams & bigrams)
5. **Model Training**
   - Logistic Regression
6. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC Curve (Micro & Macro Average)

---

##  Model Performance Metrics
- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1-score
- Specificity
- MCC
- ROC-AUC

---

##  Visualizations Included
- Word Clouds
- Dataset size comparison
- TF-IDF feature importance
- PCA projection
- Confusion Matrix
- ROC Curves

---

##  Saved Files
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `cyberbullying_lr_model.pkl`
- `preprocessed_df.csv`

---

##  How to Run
1. Install required libraries:
   ```bash
   pip install numpy pandas nltk scikit-learn matplotlib seaborn wordcloud joblib
2. Run the Python script:
   ```bash
   python cyberbullying_detection_logistic_regression.py
Conclusion

This project demonstrates how traditional machine learning techniques combined with NLP can effectively identify cyberbullying content and support online safety initiatives.
