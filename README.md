# Codsoft_data_science
Internship projects

Project 1 Iris Flower Classifier – Streamlit App
This is a simple and interactive machine learning web app built using Streamlit and scikit-learn. It allows users to predict the species of an Iris flower based on its sepal and petal measurements using a Random Forest Classifier.

🚀 Features
Uses the built-in Iris dataset from scikit-learn.

Built with Streamlit for a clean, interactive UI.

Users can input:

Sepal length

Sepal width

Petal length

Petal width

Displays the predicted species: Setosa, Versicolor, or Virginica.

🧠 ML Model
Algorithm: Random Forest Classifier

Library: scikit-learn

Trained on the complete dataset for demonstration purposes.
![image](https://github.com/user-attachments/assets/5c7cec02-7471-4354-9530-750723587927)

Project 2 
# 🎬 IMDB Sentiment Analysis

A Machine Learning/NLP project that classifies movie reviews from the IMDB dataset as **positive** or **negative** based on their content using deep learning models.

---

## 📌 Project Overview

This project focuses on building a sentiment analysis model using the IMDB dataset, a popular benchmark for natural language processing. It utilizes preprocessing techniques and deep learning (RNN/LSTM/CNN) to classify the sentiment of movie reviews.

---

## 💡 Features

- Cleaned and preprocessed textual data
- Tokenization, Padding, and Embedding layers
- Deep learning model for binary classification (Positive / Negative)
- Accuracy and loss visualization
- Predict custom movie reviews

---

## 📁 Dataset

- **Source**: [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- Contains 50,000 labeled reviews split equally into training and testing sets.
- Balanced dataset with 25,000 positive and 25,000 negative reviews.

---

## ⚙️ Tech Stack

- **Language**: Python
- **Libraries**:
  - TensorFlow / Keras
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
  - NLTK (for optional text preprocessing)

---

## 🧠 Model Architecture

> Example: Using LSTM

- Embedding Layer

- Dropout
- Dense (Sigmoid activation for binary output)

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
![image](https://github.com/user-attachments/assets/f9745467-36d6-4efe-a94a-142ea080226d)

Project 3 :

🚢 Titanic Survival Prediction with Streamlit
This project uses machine learning to predict whether a passenger survived the Titanic disaster based on features like age, ticket class, sex, and more. It also includes an interactive Streamlit web app where users can input passenger details and get instant survival predictions.

📌 Project Features
Predicts survival using a Random Forest Classifier

Interactive Streamlit app for real-time predictions

Clean preprocessing pipeline using ColumnTransformer

Feature inputs include:

Passenger class (Pclass)

Sex

Age

Siblings/Spouses aboard

Parents/Children aboard

Fare paid

Embarkation port

🧠 Tech Stack
Component	Tools Used
Language	Python
ML Library	scikit-learn, pandas, numpy
Web Framework	Streamlit
Visualization	Streamlit widgets + prediction text
Model Storage	joblib!
[image](https://github.com/user-attachments/assets/8643c4bf-abdd-4a3f-a268-266800c210d9)

