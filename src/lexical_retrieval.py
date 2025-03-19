import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk
from utils import search_queries


# Preprocess the passages
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to search for relevant passages
def search(query, vectorizer, passages, tfidf_matrix, top_k=5):
    # Preprocess the query
    preprocessed_query = preprocess(query)
    # Transform the query to TF-IDF vector
    query_vector = vectorizer.transform([preprocessed_query])
    # Calculate similarity scores
    similarity_scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()
    # Get the indices of the top k passages
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    # Return the top passages and their scores
    results = []
    for idx in top_indices:
        results.append({"passage": passages[idx], "score": similarity_scores[idx]})
    return results


def preprocess_text(text):
    passages = sent_tokenize(text)
    preprocessed_passages = list(set([preprocess(passage) for passage in passages]))
    return preprocessed_passages


def vectorize(text, vectorizer, return_passages=False):
    preprocessed_passages = preprocess_text(text)
    # Create TF-IDF vectors
    vectorizer.fit(preprocessed_passages)
    if return_passages:
        return vectorizer, preprocessed_passages
    return vectorizer


    # Example: Search for a query and display the top 5 passages

# Download necessary NLTK data
if __name__ == "__main__":
    print("Downloading NLTK data")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    # Load the text file
    with open("harry_potter.txt", "r", encoding="utf-8") as file:
        text = file.read()

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer = vectorize(text, vectorizer)
    tfidf_matrix = vectorizer.transform(preprocess_text(text))
    QUERY1 = "magic wand"
    QUERY2 = "Harry be careful!"
    QUERY3 = "Voldemort is here"
    search_queries(
        queries=[QUERY1, QUERY2, QUERY3],
        search_fn=search,
        top_k=5,
        vectorizer=vectorizer,
        passages=preprocess_text(text),
        tfidf_matrix=tfidf_matrix,
    )
