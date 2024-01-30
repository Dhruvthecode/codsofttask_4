import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv(r"C:\Users\Dhruv Sawant\Documents\spam.csv", encoding='latin-1')

# Assuming 'v1' column contains 'spam' or 'ham' and 'v2' column contains the SMS content
if 'v1' in data.columns and 'v2' in data.columns:
    data = data[['v1', 'v2']]
else:
    print("Columns 'v1' and 'v2' not found in the dataset.")

# Rename columns for clarity
data.columns = ['label', 'message']
# Split the data into features (X) and target variable (y)
X = data['message']
y = data['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary (0 for ham, 1 for spam)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features parameter
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Choose classifiers (Naive Bayes, Logistic Regression, SVM)
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True)
}
# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    # Train and evaluate models with plotting
for clf_name, clf in classifiers.items():
    print(f'\nTraining and evaluating {clf_name}...')
    
    # Train the model
    clf.fit(X_train_tfidf, y_train)
 # Make predictions
y_pred = clf.predict(X_test_tfidf)
y_prob = clf.predict_proba(X_test_tfidf)[:, 1]  # Probability estimates for the positive class
    # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)  
print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{report}') 
 # Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, title=f'Confusion Matrix - {clf_name}')
    
    # Plot ROC curve
plot_roc_curve(y_test, y_prob, title=f'ROC Curve - {clf_name}')
for clf_name, clf in classifiers.items():
    print(f'\nResults for {clf_name}:')
    
    # Make predictions
    y_pred = clf.predict(X_test_tfidf)
    y_prob = clf.predict_proba(X_test_tfidf)[:, 1]  # Probability estimates for the positive class
    
    # Display predictions and probability estimates for the first few examples
    for i in range(5):  # Displaying results for the first 5 examples
        print(f"Message: {X_test.iloc[i]}")
        print(f"Actual Label: {y_test.iloc[i]}, Predicted Label: {y_pred[i]}, Probability: {y_prob[i]:.4f}\n")
