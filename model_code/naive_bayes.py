import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import json

# Step 1: Load preprocessed data
def load_data(pickle_path, csv_path):
    """
    Load tokenized text data from a pickle file and sentiment labels from CSV.
    """
    # Load tokenized text data
    with open(pickle_path, "rb") as file:
        data_by_words = pickle.load(file)
        
    # Load CSV file for labels
    headers = ["target", "id", "date", "flag", "user", "text"]
    text_df = pd.read_csv(csv_path, names=headers)
    text_df = text_df.reset_index(drop=True)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(text_df['target'])
    
    return data_by_words, labels

# Step 3: Feature Extraction
def extract_features(train_texts, test_texts):
    """
    Convert tokenized text data into numerical features using TF-IDF.
    """
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    
    # Transform text to count vectors and then to TF-IDF
    train_counts = vectorizer.fit_transform(train_texts)
    train_tfidf = tfidf_transformer.fit_transform(train_counts)
    
    test_counts = vectorizer.transform(test_texts)
    test_tfidf = tfidf_transformer.transform(test_counts)
    
    return train_tfidf, test_tfidf

# Step 4: Split Data
def split_data(texts, labels):
    """
    Split data into training and testing sets.
    """
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes Model
def train_model(train_features, train_labels):
    """
    Train a Naive Bayes classifier on the training features and labels.
    """
    model = MultinomialNB()
    model.fit(train_features, train_labels)
    return model

# Step 6: Make Predictions
def make_predictions(model, test_features):
    """
    Use the trained model to make predictions on test data.
    """
    return model.predict(test_features)

# Step 7: Evaluate Model
def evaluate_model(predictions, true_labels):
    """
    Evaluate model performance using accuracy and classification report.
    """
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    print(f'Accuracy: {accuracy}')
    print(report)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def preprocess_and_vectorize(train_texts, test_texts, config):
    """
    Create a term-document matrix based on the configuration settings.
    """
    # Initialize the vectorizer
    vectorizer = CountVectorizer(
        max_features=config.get("max_features"),
        ngram_range=tuple(config.get("ngram_range", (1, 1)))
    )
    
    # Fit and transform training data
    train_counts = vectorizer.fit_transform(train_texts)
    
    # Transform test data
    test_counts = vectorizer.transform(test_texts)
    
    # Apply TF-IDF transformation if specified
    if config.get("use_tfidf", False):
        tfidf_transformer = TfidfTransformer()
        train_features = tfidf_transformer.fit_transform(train_counts)
        test_features = tfidf_transformer.transform(test_counts)
    else:
        train_features = train_counts
        test_features = test_counts
    
    return train_features, test_features

def run_ablation_study(train_texts, train_labels, test_texts, test_labels, configs):
    """
    Run the ablation study by iterating over each configuration.
    """
    results = []
    
    for config in configs:
        # Preprocess and vectorize based on the current config
        train_features, test_features = preprocess_and_vectorize(train_texts, test_texts, config)
        
        # Train Naive Bayes model
        model = MultinomialNB()
        model.fit(train_features, train_labels)
        
        # Make predictions and evaluate
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        # Store results for the current config
        results.append({
            "config_name": config["name"],
            "accuracy": accuracy,
            "report": report
        })
        
        print(f"Configuration: {config['name']}")
        print(f"Accuracy: {accuracy}")
        print(classification_report(test_labels, predictions))
        
    return results

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import json
import numpy as np

# Load configurations from JSON
def load_configurations(config_path="bayes_config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config["configurations"]

# Convert tokenized words to text
def tokens_to_text(data_by_words):
    return [" ".join(words) for words in data_by_words]

# Feature extraction based on configuration
def preprocess_and_vectorize(train_texts, config):
    vectorizer = CountVectorizer(
        max_features=config.get("max_features"),
        ngram_range=tuple(config.get("ngram_range", (1, 1)))
    )
    train_counts = vectorizer.fit_transform(train_texts)

    # Apply TF-IDF if specified
    if config.get("use_tfidf", False):
        tfidf_transformer = TfidfTransformer()
        train_features = tfidf_transformer.fit_transform(train_counts)
    else:
        train_features = train_counts

    return train_features, vectorizer

# Ablation study with k-fold cross-validation
def run_kfold_ablation_study(texts, labels, configs, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for config in configs:
        print(f"Running configuration: {config['name']}")
        
        fold_accuracies = []
        fold_reports = []
        
        for train_index, test_index in skf.split(texts, labels):
            # Split data for current fold
            train_texts, test_texts = texts[train_index], texts[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            # Preprocess and vectorize based on the current config
            train_features, vectorizer = preprocess_and_vectorize(train_texts, config)
            test_counts = vectorizer.transform(test_texts)
            test_features = test_counts if not config.get("use_tfidf", False) else TfidfTransformer().fit_transform(test_counts)

            # Train Naive Bayes model on current fold
            model = MultinomialNB()
            model.fit(train_features, train_labels)
            
            # Predict and evaluate
            predictions = model.predict(test_features)
            accuracy = accuracy_score(test_labels, predictions)
            report = classification_report(test_labels, predictions, output_dict=True)

            fold_accuracies.append(accuracy)
            fold_reports.append(report)
        
        # Calculate average performance across folds for the current config
        avg_accuracy = np.mean(fold_accuracies)
        avg_report = {
            metric: np.mean([fold[metric]['f1-score'] for fold in fold_reports])
            for metric in fold_reports[0].keys() if metric != 'accuracy'
        }
        
        # Store results
        results.append({
            "config_name": config["name"],
            "average_accuracy": avg_accuracy,
            "average_f1_scores": avg_report
        })

        print(f"Config {config['name']}: Average Accuracy: {avg_accuracy}")
        print(f"Average F1 Scores: {avg_report}")
        
    return results

# Example main function to run the k-fold ablation study
def main(pickle_path, csv_path):
    # Load and preprocess data
    data_by_words, labels = load_data(pickle_path, csv_path)
    texts = np.array(tokens_to_text(data_by_words))  # Convert to NumPy array for indexing
    labels = np.array(labels)

    # Load configurations
    configs = load_configurations()
    
    # Run k-fold ablation study
    results = run_kfold_ablation_study(texts, labels, configs, k=5)
    
    # Save results to a JSON file for analysis
    with open("kfold_ablation_results.json", "w") as file:
        json.dump(results, file, indent=4)


# Example usage:
main("sam_data_by_words.pkl", "LabeledTweets.csv")



