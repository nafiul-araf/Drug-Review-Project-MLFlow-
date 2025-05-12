
from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
# from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import re
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')



def transform_text(text):
    "Transform the text by text lowering, tokenizing, removing special characters, stopwords, punctuations and stemming."
    
    # Lower case the text
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove special characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming the texts
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]

    # Join the tokens back into a single string
    return ' '.join(text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/nafiul-araf/Drug-Review-Project-MLFlow-.mlflow')

dagshub.init(repo_owner='nafiul-araf', repo_name='Drug-Review-Project-MLFlow-', mlflow=True)
# -------------------------------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
# registry = CollectorRegistry()

# # Define your custom metrics using this registry
# REQUEST_COUNT = Counter(
#     "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
# )
# REQUEST_LATENCY = Histogram(
#     "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
# )
# PREDICTION_COUNT = Counter(
#     "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
# )

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "naive_bayes"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Routes
@app.route("/")
def home():
    # REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    # REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    # REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    # Clean text
    text = transform_text(text)
    # Convert to features
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    result = model.predict(features_df)
    prediction = result[0]

    # Increment prediction count metric
    # PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    # REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

# @app.route("/metrics", methods=["GET"])
# def metrics():
#     """Expose only custom Prometheus metrics."""
#     return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker