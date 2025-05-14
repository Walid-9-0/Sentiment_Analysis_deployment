import streamlit as st
import pickle
import pandas as pd
import io
import matplotlib.pyplot as plt

# إعداد الصفحة
st.set_page_config("Sentiment Classifier", "💬", layout="centered")

# Load Model and TF_IDF
@st.cache_resource
def load_model():
    with open("log_model_tfidf.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# labels
label_map = {
    0: ("Negative", "😠", "red"),
    1: ("Positive", "😊", "green")
}

# CSS background
st.markdown("""
<style>
.stApp {
    background-image: url("https://3sixfive.co.uk/wp-content/uploads/2022/05/sentiment-analysis-blog-header-scaled.jpg");
    background-size: cover;
    background-position: center;
}
.content-box {
    background-color: rgba(255,255,255,0.85);
    padding: 2rem;
    border-radius: 15px;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.title("💬 Sentiment Classifier")
st.markdown("Enter a sentence or upload a file to classify it as **Positive** or **Negative**.")

# session_state history
if "history" not in st.session_state:
    st.session_state.history = []

# input
input_option = st.radio("Choose input type:", ["Text Input", "Upload File"])

if input_option == "Text Input":
    text = st.text_area("📝 Enter text", height=100)
    if st.button("🔍 Classify") and text.strip():
        X = tfidf.transform([text])
        label = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        conf = max(prob) * 100

        sentiment, emoji, color = label_map.get(label, ("Unknown", "❓", "gray"))
        st.markdown(f"""
            <div style='background:{color};color:white;padding:1rem;border-radius:10px;text-align:center;font-weight:bold'>
                {emoji} {sentiment} ({conf:.2f}%)
            </div>
        """, unsafe_allow_html=True)

        st.session_state.history.append({
            "Text": text, "Prediction": sentiment, "Confidence (%)": f"{conf:.2f}"
        })

elif input_option == "Upload File":
    uploaded_file = st.file_uploader("📎 Upload CSV / Excel / TXT file", type=["csv", "xlsx", "txt"])
    text_column = st.text_input("Column name containing text", value="Text")

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                 df = pd.read_csv(uploaded_file, encoding="latin1")
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["Text"])

        if text_column not in df.columns:
            st.warning(f"⚠️ Column '{text_column}' not found.")
        else:
            texts = df[text_column].dropna().astype(str).tolist()
            X = tfidf.transform(texts)
            labels = model.predict(X)
            probs = model.predict_proba(X)

            for text, label, prob in zip(texts, labels, probs):
                conf = max(prob) * 100
                sentiment, _, _ = label_map[label]
                st.session_state.history.append({
                    "Text": text, "Prediction": sentiment, "Confidence (%)": f"{conf:.2f}"
                })

            st.success(f"✅ Classified {len(texts)} rows.")

# show history
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.subheader("📚 Classification History")
    st.dataframe(df_history, use_container_width=True)

    # Pie Chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df_history["Prediction"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "red"], startangle=90)
    ax1.set_title("Sentiment Distribution (Pie Chart)")
    st.pyplot(fig1)


    st.write(sentiment_counts)


    # download CSV
    csv_buffer = io.StringIO()
    df_history.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="sentiment_results.csv",
        mime="text/csv"
    )

# Clear History
if st.button("🗑️ Clear History"):
    st.session_state.history = []

st.markdown('</div>', unsafe_allow_html=True)
