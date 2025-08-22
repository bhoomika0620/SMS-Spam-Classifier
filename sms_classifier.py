# SMS Spam Classifier

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 2. Load dataset
# Example dataset: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# Dataset should have two columns: ['label', 'message']
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label', 'message']

# 3. Encode labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham':0, 'spam':1})

# 4. Split data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Model training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Predictions
y_pred = model.predict(X_test_vec)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Test on custom message
test_message = ["Congratulations! You've won a $1000 Walmart gift card. Call now!",
                "Hey, are we still meeting at 6pm today?"]
test_message_vec = vectorizer.transform(test_message)
print(model.predict(test_message_vec))  # 1=spam, 0=ham
