import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("Resume/Resume.csv", encoding="latin1")

# Drop missing values
df = df.dropna(subset=['Resume_str', 'Category'])

# Features and labels
X = df['Resume_str']
y = df['Category']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# Try predicting a sample resume's job category
sample = ["Experienced software developer skilled in Python, machine learning, and data analysis"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("\nSample Resume Predicted Category:", prediction[0])
