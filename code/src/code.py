import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
import os
from datetime import datetime

# Create a Flask app instance
app = Flask(__name__)

# Initialize the logger with size-based rotation
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

log_file = "user_recommendations.log"
handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)  # 1 MB max file size, keep 5 backups
handler.setFormatter(log_formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger()  # Root logger
logger.setLevel(logging.INFO)  # Set the logging level
logger.addHandler(handler)

# Expanded sample dataset with more user IDs
data = {
    "userID": ["U001", "U002", "U003", "U004", "U005", "U006", "U007", "U008", "U009", "U010"],
    "transaction_history": [
        "Stocks, Mutual Funds, Savings",
        "Bonds, Real Estate, Insurance",
        "Fixed Deposits, Savings, Credit Card",
        "Cryptocurrency, Stocks, Commodities",
        "Savings, Mutual Funds, Bonds",
        "Real Estate, Savings, Insurance",
        "Stocks, Cryptocurrency, Mutual Funds",
        "Fixed Deposits, Emergency Savings, Bonds",
        "Savings, Commodities, Real Estate",
        "Mutual Funds, Stocks, Fixed Deposits"
    ],
    "income": [50000, 75000, 60000, 90000, 45000, 80000, 65000, 72000, 47000, 89000],
    "risk_profile": [
        "Conservative", "Moderate", "Aggressive", "Aggressive", "Conservative",
        "Moderate", "Aggressive", "Conservative", "Moderate", "Moderate"
    ],
    "goal": [
        "Retirement", "Buying a House", "Education", "Investments", "Emergency Savings",
        "Retirement", "Investments", "Buying a House", "Emergency Savings", "Education"
    ],
    "city": [
        "Mumbai", "New York", "London", "Tokyo", "Singapore",
        "Sydney", "Berlin", "Paris", "Dubai", "Toronto"
    ],
    "market_trend": [
        "Bullish", "Bearish", "Neutral", "Bullish", "Bearish",
        "Neutral", "Bullish", "Bearish", "Neutral", "Bullish"
    ],
    "embeddings": [
        [0.2, 0.5, 0.8], [0.6, 0.1, 0.3], [0.4, 0.7, 0.9],
        [0.5, 0.4, 0.6], [0.3, 0.8, 0.2], [0.7, 0.2, 0.5],
        [0.3, 0.6, 0.8], [0.5, 0.7, 0.9], [0.4, 0.3, 0.7],
        [0.6, 0.8, 0.4]
    ],
    "product_name": [
        "Pension Fund A", "Real Estate B", "Education Loan C",
        "Crypto Investment Plan", "Mutual Fund D", "Retirement Fund E",
        "Global Equity Fund", "Bond Investment Plan", "Savings Growth Fund", "Education Trust Fund"
    ],
    "price": [10000, 500000, 20000, 70000, 15000, 30000, 80000, 45000, 12000, 25000],
    "minimum_income": [20000, 60000, 40000, 80000, 30000, 50000, 70000, 40000, 35000, 45000]
}

df = pd.DataFrame(data)

# Step 1: Fetch market trends dynamically using Alpha Vantage
def fetch_market_trend():
    """Fetch real-time market trend using Alpha Vantage."""
    try:
        api_key = "P97PXYG6EAJ48R8S"  # Replace with your Alpha Vantage API key
        symbol = "SPY"  # Using the S&P 500 ETF as an example for market trend
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=15min&apikey={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        # Extract the latest closing price and determine the trend
        time_series = data.get("Time Series (15min)", {})
        latest_timestamp = next(iter(time_series.keys()), None)
        if latest_timestamp:
            close_price = float(time_series[latest_timestamp]["4. close"])
            # Simplistic trend analysis: If the price has increased, return "Bullish"
            return "Bullish" if close_price > 400 else "Bearish"  # Adjust threshold as needed
        return "Neutral"
    except Exception as e:
        print(f"Error fetching market trend: {e}")
        return "Unknown"

# Step 2: Fetch financial news (mocked for now)
def fetch_financial_news():
    """Mock function to fetch financial news headlines."""
    return [
        "Stock market rallies as inflation slows down",
        "Tech stocks suffer amid regulatory concerns",
        "Investors optimistic about renewable energy projects",
        "Global recession fears increase as central banks hike rates"
    ]

# Step 3: Analyze Sentiment with TextBlob and VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER."""
    scores = analyzer.polarity_scores(text)
    compound_score = scores["compound"]
    sentiment = "positive" if compound_score > 0 else "negative" if compound_score < 0 else "neutral"
    return sentiment, compound_score

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
    return sentiment, polarity

print("Script is starting...")


# Step 4: Assess Sentiment for Products
def assess_product_sentiment(product_name):
    """Analyze sentiment of news related to a specific financial product."""
    news_headlines = fetch_financial_news()
    sentiment_scores = []

    for headline in news_headlines:
        if product_name.lower() in headline.lower():  # Check if product is mentioned
            sentiment, score = analyze_sentiment_vader(headline)
            sentiment_scores.append(score)

    return np.mean(sentiment_scores) if sentiment_scores else 0.0

# Step 5: Financial Education Recommendations
educational_materials = {
    "Retirement": "Top 10 ways to plan for retirement effectively.",
    "Buying a House": "A guide to saving for your first home.",
    "Education": "How to fund higher education efficiently.",
    "Investments": "Introduction to mutual funds and stock markets.",
    "Emergency Savings": "Building a reliable emergency fund."
}

def get_educational_material(goal):
    """Provide educational resources based on user goals."""
    return educational_materials.get(goal, "Learn more about financial planning!")

# Step 6: Log Historical Recommendations
def log_recommendations(user_id, recommendations):
    """Log user recommendations with date and timestamp."""
    # Get the current date and time Example: 2025-03-25 15:45:12
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format date and time
    log_message = f"[{current_time}] User {user_id}: {recommendations}"
    logger.info(log_message)

# Step 7: Recommendation function with Alpha Vantage and sentiment integration
def recommend_financial_products(user_id, city, market_trend, risk_profile, goal, budget, top_n=3):
    """Recommend financial products based on user preferences and dynamic context."""
    # Filter user-specific data
    user_data = df[df["userID"] == user_id]
    if user_data.empty:
        return {"error": "No data available for this user."}

    # Retrieve embeddings
    user_embeddings = np.array(list(user_data["embeddings"])).reshape(1, -1)
    all_embeddings = np.array(list(df["embeddings"]))

    # Compute similarity scores
    similarity_scores = cosine_similarity(user_embeddings, all_embeddings)

    # Dynamic scoring for context
    df["goal_match"] = df["goal"].apply(lambda x: 1 if x.lower() == goal.lower() else 0)
    df["market_match"] = df["market_trend"].apply(lambda x: 1 if x.lower() == market_trend.lower() else 0)
    df["risk_match"] = df["risk_profile"].apply(lambda x: 1 if x.lower() == risk_profile.lower() else 0)
    df["budget_match"] = df["price"].apply(lambda x: 1 if x <= budget else 0)
    df["loan_eligibility"] = df["minimum_income"].apply(lambda x: 1 if x <= user_data["income"].values[0] else 0)

    # Real-Time Risk Assessment and Sentiment Scoring
    df["sentiment_score"] = df["product_name"].apply(assess_product_sentiment)

    # Combine similarity and contextual scores
    contextual_scores = (
        df["goal_match"] + df["market_match"] + df["risk_match"] +
        df["budget_match"] + df["loan_eligibility"] + df["sentiment_score"]
    )
    weighted_scores = similarity_scores[0] + contextual_scores.values

    # Get top N recommendations
    top_indices = weighted_scores.argsort()[-top_n:][::-1]  # Sort in descending order
    recommendations = df.iloc[top_indices][["product_name", "goal", "price"]].to_dict(orient="records")

    # Log recommendations
    log_recommendations(user_id, recommendations)

    # Educational Material
    education_resource = get_educational_material(goal)

    # Fetch real-time stock price (Alpha Vantage example)
    stock_price = fetch_market_trend()  # Example: fetch stock data or market trend

    # Return recommendations and additional insights
    return {
        "recommendations": recommendations,
        "goal_specific_advice": education_resource,
        "market_trend": stock_price,
        "diversification_plan": {
            "stocks": 0.4 if risk_profile == "Aggressive" else 0.2,
            "bonds": 0.3 if risk_profile == "Conservative" else 0.2,
            "savings": 0.3
        }
    }

# Step 8: Create API Endpoint
@app.route('/recommend', methods=['GET'])
def recommend():
    """API endpoint to recommend financial products."""
    try:
        # Debugging: Print received parameters
        print("Received request with parameters:")
        print(f"userID: {request.args.get('userID')}")
        print(f"city: {request.args.get('city')}")
        print(f"goal: {request.args.get('goal')}")
        print(f"risk_profile: {request.args.get('risk_profile')}")
        print(f"budget: {request.args.get('budget')}")

        # Get query parameters
        user_id = request.args.get("userID", "U001")  # Default to "U001" if not provided
        city = request.args.get("city", "Mumbai")  # Default to "Mumbai" if not provided
        goal = request.args.get("goal", "Retirement")  # Default to "Retirement" if not provided
        risk_profile = request.args.get("risk_profile", "Conservative")  # Default to "Conservative"
        budget = float(request.args.get("budget", 100000))  # Default to 100,000

        # Fetch real-time market trend
        print("Fetching market trend...")
        market_trend = fetch_market_trend()

        # Generate recommendations
        print("Generating recommendations...")
        response = recommend_financial_products(user_id, city, market_trend, risk_profile, goal, budget)
        print("Recommendations generated:", response)

        # Ensure a valid response
        return jsonify(response)

    except Exception as e:
        # Log the exception and return an error response
        # print(f"Error in /recommend: {e}")
        logger.error(f"Error in /recommend: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
