import utils.clients
import utils.http_session
import utils.transport
import utils.session
import os
import pickle
from flask import Flask, request, render_template
from news_fetcher import smart_search_news, get_api_call_count

app = Flask(__name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"))

vector = None
model = None

def load_models():
    global vector, model
    if vector is not None and model is not None:
        return True
    try:
        import joblib
        v_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
        m_path = os.path.join(os.path.dirname(__file__), "fake_news_model.pkl")
        vector = joblib.load(v_path)
        model = joblib.load(m_path)
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    news = ""
    prediction_text = ""
    ml_result = None
    news_results = []
    api_calls = get_api_call_count()
    news_error = None
    search_query = ""
    search_label = ""

    if request.method == "POST":
        news = request.form.get("news", "").strip()
        if not news:
            return render_template(
                "prediction.html",
                prediction_text="Please enter a news headline for prediction.",
            )

        models_loaded = load_models()
        if models_loaded:
            try:
                predict = model.predict(vector.transform([news]))
                label = predict[0]
                proba = None
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(vector.transform([news]))
                    proba = abs(scores[0])
                ml_result = {"label": label, "confidence": proba}
            except Exception as e:
                prediction_text = f"ML Model error: {str(e)}"

        news_data = smart_search_news(news)
        if "error" in news_data and news_data["error"]:
            news_error = news_data["error"]
        news_results = news_data.get("results", [])
        search_query = news_data.get("query_used", news)
        search_label = news_data.get("query_label", "full headline")
        api_calls = get_api_call_count()

        if ml_result:
            label = ml_result["label"]
            if news_results and label == "FAKE":
                prediction_text = (
                    f"ML says FAKE, but {len(news_results)} matching live news articles found. "
                    "The news might be real. See matching articles below."
                )
            elif news_results and label == "REAL":
                prediction_text = (
                    f"Likely REAL – ML & live news both indicate this is genuine. "
                    f"Found {len(news_results)} matching article(s)."
                )
            elif not news_results and label == "REAL":
                prediction_text = (
                    f"ML predicts REAL (no matching live news found, "
                    "but model is confident based on training data)."
                )
            else:
                prediction_text = (
                    f"ML predicts FAKE – no matching real news found online. "
                    "This headline could not be verified against live sources."
                )
        elif news_results:
            prediction_text = (
                f"ML model unavailable, but {len(news_results)} matching live news articles found. "
                "The news appears to be real based on live sources."
            )
        elif not news_results and not models_loaded:
            prediction_text = (
                "ML model unavailable and no matching live news found. "
                "Please try a different headline."
            )

    return render_template(
        "prediction.html",
        prediction_text=prediction_text,
        news=news,
        ml_result=ml_result,
        news_results=news_results,
        news_error=news_error,
        search_query=search_query,
        search_label=search_label,
        api_calls=api_calls,
    )

if __name__ == "__main__":
    app.run()
