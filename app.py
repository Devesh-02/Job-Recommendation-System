import streamlit as st
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------
# NLTK Setup
# -----------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Load Job Dataset

df = pd.read_csv("Jobs dataset.csv")
df["clean_job_title"] = df["job_title"].astype(str).apply(clean_text)
df["clean_job_description"] = df["job_description"].astype(str).apply(clean_text)

# Build TF-IDF matrix for job recommendation
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["clean_job_title"] + " " + df["clean_job_description"])

# Recommendation Functions

def recommend_jobs(skills="", location="", job_type="", top_n=5):
    # Skills filtering
    skill_vector = vectorizer.transform([clean_text(skills)])
    similarity = cosine_similarity(skill_vector, tfidf_matrix)
    df['similarity'] = similarity[0]

    # Location & Job type filtering
    filtered = df.copy()
    if location:
        filtered = filtered[filtered['city'].str.contains(location, case=False, na=False) |
                            filtered['state'].str.contains(location, case=False, na=False)]
    if job_type:
        filtered = filtered[filtered['job_type'].str.contains(job_type, case=False, na=False)]

    # Sort by similarity
    filtered = filtered.sort_values(by='similarity', ascending=False)
    return filtered[['job_title','company_name','city','state','job_type']].head(top_n)

# Streamlit UI
st.title("🇮🇳 Job Recommendation System")

skills_input = st.text_input("Enter your skills (comma-separated):", "")
location_input = st.text_input("Enter city or state:", "")
job_type_input = st.selectbox("Select job type:", ["", "Full-time", "Part-time", "Remote"])

if st.button("Find Jobs"):
    results = recommend_jobs(skills_input, location_input, job_type_input)
    if results.empty:
        st.write("No jobs found matching your criteria.")
    else:
        st.table(results)

