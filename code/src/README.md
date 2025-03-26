**Setup Instructions**
**Prerequisites**
a)	Python (3.7 or higher).
b)	Required Python libraries:
  o	pandas
  o	numpy
  o	scikit-learn
  o	Flask
  o	vaderSentiment
  o	textblob
  o	transformers
  o	requests
c)	An Alpha Vantage API key for fetching market data. (access key is required)
**Installation**
a)	Clone the repository or download the code files.
b)	Install the necessary dependencies: pip install pandas numpy scikit-learn flask vaderSentiment textblob transformers requests
c)	Place the data.xlsx file containing user data in the specified directory.

**Running the Application**
a)	Start the Flask server:
b)	Access the /recommend endpoint by sending GET requests with user-specific parameters:
c)	http://localhost:5000/recommend?userID=U001&city=Mumbai&goal=Retirement&risk_profile=Conservative&budget=100000
