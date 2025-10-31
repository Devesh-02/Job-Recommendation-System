🌟 Job Recommendation System

🚀 Live Demo: https://job-recommendation-system-demo.streamlit.app/

📂 Repository: GitHub – Devesh-02/Job-Recommendation-System

🧠 Project Overview

The Job Recommendation System is a machine learning web app that recommends suitable jobs to users based on their skills, preferred location, and job type.
It uses TF-IDF vectorization and cosine similarity to match user inputs with job descriptions from real job listings scraped from Indian job portals.

🎯 Key Features

🔍 Recommends the most relevant jobs based on user-entered skills

🏙️ Filters results by city and job type (Full-time, Part-time, Remote)

🧩 Uses TF-IDF and cosine similarity for intelligent text-based matching

💬 Clean and interactive Streamlit web interface

⚙️ Lightweight and fully deployable on Streamlit Cloud

🧾 Dataset Information

This project uses the Indeed India Job Dataset sourced from Kaggle:
🔗 https://www.kaggle.com/datasets/promptcloud/indeed-india-job-dataset

The dataset contains thousands of job postings scraped from Indeed India, including:

Job Title

Company Name

Location (City, State)

Job Type

Job Description

A smaller, cleaned version (jobs_dataset.csv) is used for faster deployment.

🧰 Tech Stack
Category	Tools Used
Programming Language	Python
Libraries	pandas, scikit-learn, nltk, streamlit
NLP Techniques	TF-IDF Vectorization, Lemmatization, Stopword Removal
Deployment	Streamlit Cloud
⚙️ Installation & Usage (Local)

Clone the repository

git clone https://github.com/Devesh-02/Job-Recommendation-System.git
cd Job-Recommendation-System


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py


Open in your browser:
👉 http://localhost:8501

🧩 How It Works

The user enters their skills, location, and job type.

The app cleans and vectorizes job descriptions using TF-IDF.

It computes cosine similarity between the user's input and job postings.

The most similar jobs are displayed as recommendations.

🙌 Acknowledgments

Dataset by PromptCloud on Kaggle

Libraries: Streamlit, Scikit-learn, NLTK, Pandas

Project developed by Devesh
