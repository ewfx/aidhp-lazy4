# Install required libraries: pip install pandas scikit-learn transformers torch openpyxl
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime 

# Step 1: Load the dataset from an Excel file
input_file = "C:\\Users\\suchi\\OneDrive\\Documents\\your_dataset.xlsx"  # Replace with your actual Excel file name
df = pd.read_excel(input_file)

# Normalize interaction scores (e.g., ratings or engagement)
scaler = MinMaxScaler()
df["interaction"] = scaler.fit_transform(df[["interaction"]])

# Add city and time information to the dataset
# Ensure your dataset has 'city' and 'time_of_day' columns

# Step 2: Generate embeddings for song lyrics using Hugging Face Transformers
# Initialize the embedding pipeline (using distilbert-base-uncased model)
embedding_pipeline = pipeline("feature-extraction", model="distilbert-base-uncased")

def generate_embeddings(text):
    """Generate embeddings for lyrics using Hugging Face."""
    embedding = embedding_pipeline(text)
    return embedding[0][0]  # Simplified output (mean pooling can be used for large vectors)

# Apply embedding generation to the 'lyrics' column
df["embeddings"] = df["lyrics"].apply(generate_embeddings)

# Step 3: Define the recommendation function
def recommend_songs(user_id, top_n=4):
    """Recommend songs to a user based on metadata and interactions."""
    user_data = df[df["userID"] == user_id]
    if user_data.empty:
        return "No data available for this user."
    
    # Get user-specific embeddings
    user_embeddings = list(user_data["embeddings"])
    
    # Compute similarity scores between user's items and all items
    all_embeddings = list(df["embeddings"])
    similarity_scores = cosine_similarity(user_embeddings, all_embeddings)
    
    # Rank songs based on similarity and return the top N recommendations
    similar_items_indices = similarity_scores.argsort(axis=1)[0][-top_n:]
    recommendations = df.iloc[similar_items_indices]["songID"].tolist()
    return recommendations

# Step 4: Test the recommendation system
# Replace 'user1' with any userID from your dataset to generate recommendations
recommended_songs = recommend_songs("c001")
print(f"Recommended songs for user1: {recommended_songs}")