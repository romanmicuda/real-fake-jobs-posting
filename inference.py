import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Define columns
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
meta_features = ['telecommuting', 'has_company_logo', 'has_questions'] + cat_cols

# Load saved components
model = joblib.load('SVC_linear.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
label_encoders = {col: joblib.load(f'label_encoder_{col}.pkl') for col in cat_cols}

def preprocess_job_posting(data, is_dict=False):
    """
    Preprocess a single job posting or a batch of postings.
    Args:
        data: Dictionary (single posting) or DataFrame (batch).
        is_dict: True if data is a dictionary, False if DataFrame.
    Returns:
        Processed feature matrix ready for prediction.
    """
    # Convert dictionary to DataFrame if needed
    if is_dict:
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Fill missing text values
    for col in text_cols:
        df[col] = df[col].fillna('')

    # Combine text columns
    df['combined_text'] = df[text_cols].agg(' '.join, axis=1)

    # Handle categorical meta-features
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        # Transform using loaded label encoder, handle unknown values
        try:
            df[col] = label_encoders[col].transform(df[col])
        except ValueError:
            # If category not seen in training, assign 'Unknown' encoding
            df[col] = label_encoders[col].transform(['Unknown'])

    # Fill missing meta-features with defaults (0 for binary, 'Unknown' for categorical)
    for col in ['telecommuting', 'has_company_logo', 'has_questions']:
        df[col] = df[col].fillna(0).astype(int)

    # TF-IDF vectorization
    X_text = tfidf.transform(df['combined_text'])

    # Combine text and meta-features
    X_meta = df[meta_features].values
    X = np.hstack((X_text.toarray(), X_meta))

    return X

def predict_job_posting(data, is_dict=False):
    """
    Predict if job posting(s) are real or fake.
    Args:
        data: Dictionary (single posting) or DataFrame (batch).
        is_dict: True if data is a dictionary, False if DataFrame.
    Returns:
        List of predictions (0 = real, 1 = fake) and probabilities (if available).
    """
    # Preprocess input
    X = preprocess_job_posting(data, is_dict)

    # Predict
    predictions = model.predict(X)

    # LinearSVC doesn't provide probabilities directly; use decision function for confidence
    confidences = model.decision_function(X)

    # Convert to probability-like scores (sigmoid normalization)
    probs = 1 / (1 + np.exp(-confidences))

    return predictions, probs

# Example usage: Single job posting (dictionary)
single_posting = {
    'title': 'Work From Home Data Entry',
    'company_profile': '',
    'description': 'Earn money from home with flexible hours. No experience needed. Apply now!',
    'requirements': 'Basic computer skills. Must have internet.',
    'benefits': 'Flexible schedule, work from home.',
    'telecommuting': 1,
    'has_company_logo': 0,
    'has_questions': 0,
    'employment_type': 'Part-time',
    'required_experience': 'Entry level',
    'required_education': 'High School or equivalent',
    'industry': 'Unknown',
    'function': 'Administrative'
}

# Predict single posting
pred, prob = predict_job_posting(single_posting, is_dict=True)
print("\nSingle Posting Prediction:")
print(f"Prediction: {'Fake' if pred[0] == 1 else 'Real'}")
print(f"Confidence Score: {prob[0]:.4f}")