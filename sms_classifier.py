import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\Om Shah\\Desktop\\ANN\\spam.csv",
                   encoding="latin-1")[["Category", "Message"]]
data.columns = ["label", "message"]

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_counts, y_train)
y_pred = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
