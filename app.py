import os
import re
import html
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="TruthLens | Smart Fake News Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "combined_news.csv")
DB_PATH = os.path.join(BASE_DIR, "truthlens_history.db")

# Label mapping used during training: 0 = Fake, 1 = Real.
LABELS = {0: "FAKE NEWS", 1: "REAL NEWS"}

SENSATIONAL_TERMS = [
    "shocking", "exposed", "secret", "breaking", "urgent", "爆",
    "scandal", "unbelievable", "you won't believe", "must see",
    "miracle", "爆発", "hidden truth", "cover up", "cover-up",
    "they don't want you to know", "jaw-dropping", "terrifying",
    "destroyed", "爆発的", "exclusive", "banned", "viral"
]

CLICKBAIT_PATTERNS = [
    r"\byou won't believe\b",
    r"\bwhat happened next\b",
    r"\bthis is why\b",
    r"\bmust see\b",
    r"\bshocking\b",
    r"\bsecret(s)? they don't want you to know\b",
    r"\bthe truth about\b",
    r"\bnumber \d+ will\b",
    r"\bhere's what happened\b",
    r"\bclick here\b",
    r"\bwatch until the end\b",
    r"\bone weird trick\b"
]

POSITIVE_WORDS = {
    "good", "great", "excellent", "success", "successful", "benefit",
    "beneficial", "positive", "improve", "improved", "improvement",
    "hope", "peace", "safe", "safety", "support", "win", "growth",
    "progress", "happy", "strong", "effective", "recovery"
}
NEGATIVE_WORDS = {
    "bad", "terrible", "shocking", "crisis", "crash", "danger",
    "dangerous", "death", "dead", "attack", "war", "fear", "fraud",
    "fake", "scam", "corrupt", "failure", "fail", "threat", "threaten",
    "violence", "violent", "disaster", "harm", "harmful", "illness",
    "disease", "victim", "criminal", "worst", "lie", "lies"
}


# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #0a1024;
    color: #edf2ff;
}
[data-testid="stHeader"] {
    background: rgba(10, 16, 36, 0.92);
}
[data-testid="stSidebar"] {
    background: #111a35;
    border-right: 1px solid #293657;
}
[data-testid="stSidebar"] * {
    color: #dce5fb;
}
.block-container {
    padding-top: 1.7rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #f4f7ff !important;
    letter-spacing: -0.5px;
}
p, label, .stCaption {
    color: #c1cce3;
}
div[data-testid="stTextArea"] textarea {
    background: #0b1229;
    color: #f2f5ff;
    border: 1px solid #596887;
    border-radius: 12px;
    min-height: 290px;
}
div[data-testid="stTextInput"] input {
    background: #0b1229;
    color: #f2f5ff;
    border: 1px solid #596887;
    border-radius: 10px;
}
.stButton > button {
    background: linear-gradient(100deg, #536b9d, #8798bd);
    color: white;
    border: 1px solid #aab9d9;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    width: 100%;
    box-shadow: 0 0 18px rgba(115, 143, 201, 0.22);
}
.stButton > button:hover {
    border-color: #b7eaff;
    color: white;
    box-shadow: 0 0 22px rgba(82, 197, 231, 0.3);
}
.panel {
    background: linear-gradient(145deg, #1b2747, #151f3b);
    border: 1px solid #344467;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}
.panel-title {
    color: #e7edfb;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.35px;
    margin-bottom: 12px;
}
.verdict {
    width: 142px;
    height: 142px;
    border-radius: 50%;
    margin: 8px auto 4px auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 4px double;
    background: #111a32;
    text-align: center;
}
.verdict small {
    font-size: 0.75rem;
    color: #e7edfb;
    font-weight: 700;
}
.verdict strong {
    font-size: 1.55rem;
    letter-spacing: 0.4px;
}
.fake { border-color: #ed4e59; color: #ff626b; box-shadow: 0 0 18px rgba(237,78,89,.14); }
.real { border-color: #2bc58b; color: #35d69b; box-shadow: 0 0 18px rgba(43,197,139,.14); }
.neutral { border-color: #8b9bbd; color: #cbd6ed; }
.meter {
    height: 15px;
    border-radius: 20px;
    background: #394663;
    overflow: hidden;
    border: 1px solid #566582;
}
.meter-fill {
    height: 100%;
    border-radius: 20px;
}
.indicator {
    background: #111b35;
    border: 1px solid #344467;
    border-radius: 10px;
    padding: 12px;
    min-height: 88px;
    margin-bottom: 8px;
}
.indicator b { color: #f1f5ff; font-size: .86rem; }
.indicator span { color: #bbc8e1; font-size: .78rem; }
.word-highlight {
    background: rgba(239, 83, 91, .22);
    color: #ff9ca2;
    border: 1px solid rgba(239,83,91,.5);
    border-radius: 4px;
    padding: 1px 3px;
    font-weight: 700;
}
.small-muted { color: #9eacc8; font-size: .78rem; }
hr { border-color: #2b3858; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# MODEL AND DATA LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_news_dataset():
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    data = pd.read_csv(DATASET_PATH)
    if "content" not in data.columns:
        title = data["title"].fillna("") if "title" in data.columns else ""
        text = data["text"].fillna("") if "text" in data.columns else ""
        data["content"] = (title + " " + text).str.strip()
    return data


@st.cache_resource
def load_similarity_matrix():
    data = load_news_dataset()
    if data.empty:
        return data, None
    model, vectorizer = load_model_assets()
    contents = data["content"].fillna("").astype(str)
    matrix = vectorizer.transform(contents)
    return data, matrix


# ---------------------------------------------------------
# SQLITE HISTORY
# ---------------------------------------------------------
def initialize_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                article TEXT NOT NULL,
                prediction TEXT NOT NULL,
                decision_strength REAL,
                sentiment TEXT,
                clickbait_count INTEGER,
                verdict TEXT,
                signed_decision_score REAL
            )
        """)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(analyses)").fetchall()}
        if "verdict" not in columns:
            connection.execute("ALTER TABLE analyses ADD COLUMN verdict TEXT")
        if "signed_decision_score" not in columns:
            connection.execute("ALTER TABLE analyses ADD COLUMN signed_decision_score REAL")
        connection.commit()


def save_history(article, prediction, decision_strength, sentiment, clickbait_count,
                 verdict, signed_decision_score):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            INSERT INTO analyses
            (created_at, article, prediction, decision_strength, sentiment,
             clickbait_count, verdict, signed_decision_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), article[:1200], prediction,
            decision_strength, sentiment, clickbait_count, verdict, signed_decision_score
        ))
        connection.commit()


def get_history():
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            """SELECT id, created_at, article, prediction, decision_strength,
                      sentiment, clickbait_count, verdict, signed_decision_score
               FROM analyses ORDER BY id DESC""",
            connection
        )


# ---------------------------------------------------------
# NLP ANALYSIS HELPERS
# ---------------------------------------------------------
def find_sensational_terms(text):
    lower_text = text.lower()
    found = []
    for term in SENSATIONAL_TERMS:
        if term in lower_text:
            found.append(term)
    return sorted(set(found))


def find_clickbait_patterns(text):
    found = []
    for pattern in CLICKBAIT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(pattern.replace(r"\b", "").replace("\\", ""))
    return found


SentimentAnalyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    Scores = SentimentAnalyzer.polarity_scores(text)

    if Scores["compound"] >= 0.05:
        Label = "Positive"
    elif Scores["compound"] <= -0.05:
        Label = "Negative"
    else:
        Label = "Neutral"

    return Label, Scores["pos"], Scores["neg"]


def highlight_terms(text, terms):
    safe_text = html.escape(text)
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        safe_text = pattern.sub(
            lambda match: f'<span class="word-highlight">{match.group(0)}</span>',
            safe_text
        )
    return safe_text.replace("\n", "<br>")


# Provisional thresholds, not calibrated probabilities.
VERIFICATION_MARGIN = 0.75

def interpret_decision_score(decision):
    if decision <= -VERIFICATION_MARGIN:
        return "LIKELY FAKE", "fake"
    if decision >= VERIFICATION_MARGIN:
        return "LIKELY REAL", "real"
    return "NEEDS VERIFICATION", "neutral"


def predict_article(text, model, vectorizer):
    features = vectorizer.transform([text])
    feature_count = int(features.nnz)

    if feature_count == 0:
        raise ValueError(
            "The text contains no recognised vocabulary. "
            "Please enter a meaningful news article."
        )

    prediction = int(model.predict(features)[0])
    decision = float(model.decision_function(features)[0])
    decision_strength = abs(decision)
    verdict, verdict_class = interpret_decision_score(decision)

    return (
        prediction,
        decision,
        decision_strength,
        verdict,
        verdict_class,
        feature_count
    )



def find_similar_articles(article, limit=5):
    data, matrix = load_similarity_matrix()
    if data.empty or matrix is None:
        return pd.DataFrame()

    model, vectorizer = load_model_assets()
    query_vector = vectorizer.transform([article])
    similarities = cosine_similarity(query_vector, matrix).ravel()

    top_indices = similarities.argsort()[::-1][:limit]
    results = data.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices] * 100

    if "label" in results.columns:
        results["label_name"] = results["label"].map(LABELS).fillna("Unknown")
    else:
        results["label_name"] = "Unknown"

    display_columns = []
    for column in ["title", "content", "subject", "label_name", "similarity"]:
        if column in results.columns:
            display_columns.append(column)

    return results[display_columns]


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
initialize_database()

with st.sidebar:
    st.markdown("## 🛡️ TRUTHLENS")
    st.caption("SMART FAKE NEWS ANALYZER")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["Home", "History", "API Docs", "Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("NLP • MACHINE LEARNING • TEXT ANALYSIS")
    st.caption("TruthLens v1.0")


# ---------------------------------------------------------
# HOME / ANALYSIS DASHBOARD
# ---------------------------------------------------------
if page == "Home":
    st.markdown("# AI FAKE NEWS DETECTION")
    st.markdown("## DASHBOARD")
    st.caption("Analyze news language, classification patterns and similar articles.")
    st.markdown("---")

    left, right = st.columns([1.02, 1.08], gap="medium")

    with left:
        st.markdown('<div class="panel"><div class="panel-title">INPUT NEWS ARTICLE</div>', unsafe_allow_html=True)
        article = st.text_area(
            "Paste the full news article or headline",
            placeholder="Paste a news headline or article here...",
            height=300,
            label_visibility="collapsed",
            key="article_input"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        analyze_clicked = st.button("🛡️  RUN AI ANALYSIS", use_container_width=True)

        st.caption("Paste article text for analysis. URL fetching is not enabled in this version.")

    if analyze_clicked:
        cleaned_article = article.strip()

        if len(cleaned_article) < 30:
            st.warning("Please enter at least 30 characters of news text for a useful analysis.")
            st.session_state.pop("latest_result", None)
        elif re.match(r"^https?://", cleaned_article, flags=re.IGNORECASE):
            st.warning("This version analyzes pasted article text, not webpages. Open the article and paste its text here.")
            st.session_state.pop("latest_result", None)
        else:
            try:
                model, vectorizer = load_model_assets()
                prediction, signed_decision_score, decision_strength, verdict, verdict_class, feature_count = predict_article(
                    cleaned_article, model, vectorizer
                )
                prediction_name = LABELS.get(prediction, "UNKNOWN")
                sensational_terms = find_sensational_terms(cleaned_article)
                clickbait_patterns = find_clickbait_patterns(cleaned_article)
                sentiment, positive_count, negative_count = analyze_sentiment(cleaned_article)

                result = {
                    "article": cleaned_article,
                    "prediction": prediction_name,
                    "signed_decision_score": signed_decision_score,
                    "decision_strength": decision_strength,
                    "verdict": verdict,
                    "verdict_class": verdict_class,
                    "feature_count": feature_count,
                    "sensational_terms": sensational_terms,
                    "clickbait_patterns": clickbait_patterns,
                    "sentiment": sentiment,
                    "positive_count": positive_count,
                    "negative_count": negative_count
                }
                st.session_state["latest_result"] = result

                save_history(
                    cleaned_article,
                    prediction_name,
                    decision_strength,
                    sentiment,
                    len(clickbait_patterns),
                    verdict,
                    signed_decision_score
                )
            except Exception as error:
                st.error(f"Analysis could not be completed: {error}")

    result = st.session_state.get("latest_result")

    with right:
        st.markdown('<div class="panel"><div class="panel-title">ANALYSIS RESULTS</div>', unsafe_allow_html=True)

        if result:
            verdict_class = result["verdict_class"]
            verdict_color = {"fake": "#ff626b", "real": "#35d69b", "neutral": "#cbd6ed"}[verdict_class]
            verdict_text = {"LIKELY FAKE": "FAKE", "LIKELY REAL": "REAL", "NEEDS VERIFICATION": "VERIFY"}[result["verdict"]]
            st.markdown(
                f'<div class="verdict {verdict_class}"><small>TRUTHLENS VERDICT</small>'
                f'<strong>{verdict_text}</strong></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="text-align:center;color:{verdict_color};font-weight:700;">'
                f'{result["verdict"]}</div>', unsafe_allow_html=True
            )
            st.caption(f'Model classification: {result["prediction"]}')
        else:
            st.markdown(
                '<div class="verdict neutral"><small>VERDICT</small>'
                '<strong>— —</strong></div>',
                unsafe_allow_html=True
            )
            st.caption("Your prediction will appear here after analysis.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">MODEL DECISION STRENGTH</div>', unsafe_allow_html=True)
        if result:
            st.metric("Signed SVM decision score", f'{result["signed_decision_score"]:+.4f}')
            st.caption("Negative scores point toward Fake; positive scores point toward Real. This margin is not a probability.")
            st.caption(f'Provisional verification band: between {-VERIFICATION_MARGIN:.2f} and +{VERIFICATION_MARGIN:.2f}.')
        else:
            st.caption("Run an analysis to view the SVM decision margin.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">LANGUAGE & CREDIBILITY SIGNALS</div>', unsafe_allow_html=True)
        if result:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="indicator"><b>⚠ Sensational terms</b><br>'
                    f'<span>{len(result["sensational_terms"])} detected</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="indicator"><b>◉ Sentiment</b><br>'
                    f'<span>{result["sentiment"]}</span></div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f'<div class="indicator"><b>⚑ Clickbait patterns</b><br>'
                    f'<span>{len(result["clickbait_patterns"])} detected</span></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="indicator"><b>ⓘ Publisher verification</b><br>'
                    '<span>Not available in this version</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.caption("Language indicators will appear here after analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    if result:
        st.info("The three-level verdict is a cautious interpretation of the SVM score. The ±0.75 thresholds are provisional, not calibrated probabilities, and do not independently verify facts.")
        st.metric(
            "Recognised TF-IDF features",
            result["feature_count"]
        )

        if result["feature_count"] < 26:
            st.warning(
                "Limited vocabulary match. The model may not have "
                "enough relevant information for a reliable prediction. "
                "Try providing the complete news article."
            )
        st.markdown("---")
        st.subheader("Why should I be suspicious?")
        if result["sensational_terms"]:
            st.write("**Sensational terms detected:**", ", ".join(result["sensational_terms"]))
            st.markdown(
                highlight_terms(result["article"], result["sensational_terms"]),
                unsafe_allow_html=True
            )
        else:
            st.info("No terms from the current sensational-language rule list were detected.")

        if result["clickbait_patterns"]:
            st.warning("Possible clickbait patterns: " + ", ".join(result["clickbait_patterns"]))
        else:
            st.success("No configured clickbait patterns were detected.")

        st.caption("These are language-pattern indicators, not proof that the article is true or false.")

        st.markdown("---")
        st.subheader("Similar news from your dataset")
        try:
            similar = find_similar_articles(result["article"], limit=5)
            if similar.empty:
                st.info("The news dataset is unavailable or empty.")
            else:
                for index, row in similar.iterrows():
                    title = row.get("title", "Dataset article")
                    content = row.get("content", "")
                    subject = row.get("subject", "Unknown")
                    label = row.get("label_name", "Unknown")
                    score = row.get("similarity", 0)
                    with st.expander(f"{title}  ·  {score:.1f}% text similarity"):
                        st.write(str(content)[:1200] + ("..." if len(str(content)) > 1200 else ""))
                        st.caption(f"Dataset label: {label} | Subject: {subject}")
        except Exception as error:
            st.warning(f"Similar-news search is unavailable: {error}")

        st.warning("TruthLens identifies patterns learned from its dataset. It does not independently fact-check claims. Always verify important information with reliable sources.")


# ---------------------------------------------------------
# HISTORY PAGE
# ---------------------------------------------------------
elif page == "History":
    st.markdown("# DETECTION HISTORY")
    st.caption("Previously analyzed news articles saved on this device.")
    history = get_history()

    if history.empty:
        st.info("No analysis history yet. Analyze an article from the Home page first.")
    else:
        total = len(history)
        fake_count = int((history["verdict"] == "LIKELY FAKE").sum())
        real_count = int((history["verdict"] == "LIKELY REAL").sum())
        verify_count = int((history["verdict"] == "NEEDS VERIFICATION").sum())
        a, b, c, d = st.columns(4)
        a.metric("Total analyses", total)
        b.metric("Likely fake", fake_count)
        c.metric("Likely real", real_count)
        d.metric("Needs verification", verify_count)

        st.markdown("---")
        for _, row in history.iterrows():
            preview = str(row["article"]).replace("\n", " ")
            preview = preview[:110] + ("..." if len(preview) > 110 else "")
            verdict_label = row["verdict"] if pd.notna(row["verdict"]) else row["prediction"]
            with st.expander(f'{row["created_at"]}  |  {verdict_label}  |  {preview}'):
                st.write(row["article"])
                signed_score = row["signed_decision_score"]
                signed_text = f"{signed_score:+.4f}" if pd.notna(signed_score) else "Not recorded"
                st.caption(
                    f'Model classification: {row["prediction"]} | Sentiment: {row["sentiment"]} | '
                    f'Clickbait patterns: {row["clickbait_count"]} | Signed SVM score: {signed_text}'
                )

        if st.button("Clear all history", type="secondary"):
            with sqlite3.connect(DB_PATH) as connection:
                connection.execute("DELETE FROM analyses")
                connection.commit()
            st.success("History cleared.")
            st.rerun()


# ---------------------------------------------------------
# API DOCUMENTATION PAGE
# ---------------------------------------------------------
elif page == "API Docs":
    st.markdown("# API DOCUMENTATION")
    st.caption("Current TruthLens version runs as a local Streamlit application.")
    st.info("A REST API has not been implemented in this version.")
    st.markdown("### Current analysis pipeline")
    st.code(
        "Article text\n"
        "    ↓\n"
        "Saved TF-IDF Vectorizer\n"
        "    ↓\n"
        "Balanced Linear SVM\n"
        "    ↓\n"
        "Signed SVM decision score\n"
        "    ↓\n"
        "Likely Fake / Needs Verification / Likely Real\n"
        "    ↓\n"
        "Clickbait + Sensational Language + Sentiment\n"
        "    ↓\n"
        "Similar-news search + SQLite history",
        language="text"
    )
    st.markdown("### Model files")
    st.code(
        "models/best_model.pkl\n"
        "models/tfidf_vectorizer.pkl",
        language="text"
    )


# ---------------------------------------------------------
# SETTINGS PAGE
# ---------------------------------------------------------
elif page == "Settings":
    st.markdown("# SETTINGS")
    st.caption("Project and model information.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("### Model configuration")
    st.write("**Classifier:** Balanced Linear SVM")
    st.write("**Verdict layer:** Three-level interpretation of signed SVM margin")
    st.write(f"**Provisional verification threshold:** ±{VERIFICATION_MARGIN:.2f}")
    st.write("**Feature extraction:** TF-IDF (unigrams + bigrams)")
    st.write("**Fake label:** 0")
    st.write("**Real label:** 1")

    model_exists = os.path.exists(MODEL_PATH)
    vectorizer_exists = os.path.exists(VECTORIZER_PATH)
    dataset_exists = os.path.exists(DATASET_PATH)

    st.markdown("### File status")
    st.write(f"{'✅' if model_exists else '❌'} Model file")
    st.write(f"{'✅' if vectorizer_exists else '❌'} Vectorizer file")
    st.write(f"{'✅' if dataset_exists else '❌'} Dataset file")

    st.markdown("### About")
    st.write(
        "TruthLens is an academic NLP and machine-learning project. "
        "Its predictions are based on patterns in the training dataset and "
        "must not be treated as definitive factual verification."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div class="small-muted" style="text-align:center;">'
    'TRUTHLENS · SMART FAKE NEWS ANALYZER · ACADEMIC PROJECT'
    '</div>',
    unsafe_allow_html=True
)
