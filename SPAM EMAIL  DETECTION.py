import string # Helps with removing punctuation
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # Reduces words to their root form

from sklearn.feature_extraction.text import CountVectorizer  #Converts text into numerical format for ML
from sklearn.model_selection import train_test_split # Splits data into training and testing sets
from sklearn.ensemble import RandomForestClassifier # Machine learning model for text classification

nltk.download('stopwords')

df = pd.read_csv(r'MACHINE-LEARNING--MODEL--IMPLEMENTATION/spam_ham_dataset.csv')

df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

df.info()

stemmer = PorterStemmer()
corpus = []
stopwords_set = set(stopwords.words('english'))

# Process text and build corpus
for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)  # Append processed text to corpus

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(corpus).toarray()
y = df.label_num 

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train RandomForest model
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(x_train, y_train)

# Evaluate the model
print(f"Model Accuracy: {clf.score(x_test, y_test) * 100:.2f}%")

# Process new email for classification
email_to_classify = df.text.values[10]

email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
email_text = ' '.join(email_text)

email_corpus = [email_text]

x_email = vectorizer.transform(email_corpus)

prediction = clf.predict(x_email)

print(f"Predicted Label: {prediction[0]}")
print(f"Actual Label: {df.label_num.iloc[10]}")
