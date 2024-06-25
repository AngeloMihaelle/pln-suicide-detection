import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Define constants
LABELS = ["suicida", "no_suicida"]
SUBSET_SIZE = 80000  # Use a subset for faster processing

# Load data
df = pd.read_csv("data_raw.csv")

# Ensure correct label mapping
df['label'] = df['label'].apply(lambda x: LABELS.index(x))

# Shuffle and select a subset of the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_subset = df.head(SUBSET_SIZE)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    df_subset['text'],
    df_subset['label'],
    train_size=0.8,
    random_state=42
)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Initialize Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Train the model
logreg.fit(X_train_tfidf, y_train)

# Predict on the validation set
y_pred = logreg.predict(X_val_tfidf)

# Print classification report
print(classification_report(y_val, y_pred, target_names=LABELS))
