import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# LOAD MODEL & VECTORIZER
# -----------------------------
svm_model = joblib.load(
    "C:/Users/user/Desktop/Dakshita project/NLP/New Project/svm_model.pkl"
)

vectorizer = joblib.load(
    "C:/Users/user/Desktop/Dakshita project/NLP/New Project/tfidf_vectorizer.pkl"
)

# -----------------------------
# COLOR PALETTE
# -----------------------------
color_palette = px.colors.qualitative.Pastel

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(
    """
    <style>

    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #d9a7c7 0%, #fffcdc 100%);
        background-attachment: fixed;
    }

    /* Main Title */
    h1 {
        text-align: center;
        color: #3b3b58;
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 3rem !important;
        font-weight: bold;
    }

    /* Subheadings */
    h2, h3 {
        color: #5C5470;
        font-family: 'Verdana', sans-serif;
    }

    /* Paragraph Text */
    p {
        color: #4B4453;
        font-size: 1rem;
    }

    /* Text Area */
    textarea {
        background-color: #F8F5F9 !important;
        color: #4B4453 !important;
        border-radius: 12px !important;
        border: 2px solid #C8A2C8 !important;
        padding: 12px !important;
        font-size: 16px !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Input Label */
    label {
        color: #553E6B !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #C8A2C8, #A3D2CA);
        color: #2E2E2E;
        border: none;
        border-radius: 14px;
        padding: 0.6rem 2rem;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #A3D2CA, #C8A2C8);
        color: #553E6B;
        transform: scale(1.02);
    }

    /* SUCCESS BOX - LAVENDER */
    .stSuccess {
        background-color: #E6E6FA !important;
        color: #553E6B !important;
        border-radius: 12px !important;
        padding: 15px !important;
        border: 2px solid #C8A2C8 !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }

    /* Warning Box */
    .stWarning {
        background-color: #FFF4CC !important;
        color: #7A5C00 !important;
        border-radius: 12px !important;
        border: 2px solid #FFD966 !important;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

    
    
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📰 News Category Classifier")

st.write(
    "Enter a news headline or short description to predict its category."
)

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.text_area(
    "Enter News Text:",
    placeholder="Example: Government launches new education policy..."
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("✨ Predict Category"):

    if user_input.strip() != "":

        # TF-IDF Transformation
        input_tfidf = vectorizer.transform([user_input])

        # Prediction
        prediction = svm_model.predict(input_tfidf)[0]

        # Display Prediction
        st.success(f"Predicted Category: {prediction}")

        # -----------------------------
        # CONFIDENCE SCORES
        # -----------------------------
        if hasattr(svm_model, "decision_function"):

            scores = svm_model.decision_function(input_tfidf)[0]
            categories = svm_model.classes_

            # Normalize scores
            scores = scores - scores.min()
            scores = scores / scores.sum()

            # Create dataframe
            df_scores = pd.DataFrame({
                "Category": categories,
                "Confidence": scores
            })

            # Sort values
            df_scores = df_scores.sort_values(
                by="Confidence",
                ascending=False
            )

            # Round values
            df_scores["Confidence"] = (
                df_scores["Confidence"].round(2)
            )

            # -----------------------------
            # TOP 3 PREDICTIONS
            # -----------------------------
            st.write("## 🔝 Top 3 Predictions")

            for i, row in df_scores.head(3).iterrows():
                st.write(
                    f"✅ {row['Category']} → {row['Confidence']:.2f}"
                )

            # -----------------------------
            # BAR CHART
            # -----------------------------
            st.write("## 📊 Confidence Bar Chart")

            fig_bar = px.bar(
                df_scores,
                x="Category",
                y="Confidence",
                color="Category",
                text="Confidence",
                color_discrete_sequence=color_palette
            )

            fig_bar.update_traces(
                textposition='outside',
                textfont=dict(
                    color="#553E6B",
                    size=14
                )
            )

            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#553E6B"),
                xaxis_title="Category",
                yaxis_title="Confidence Score",
                xaxis=dict(showticklabels=False)
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

            # -----------------------------
            # PIE CHART
            # -----------------------------
            st.write("## 🥧 Confidence Pie Chart")

            fig_pie = px.pie(
                df_scores,
                names="Category",
                values="Confidence",
                color_discrete_sequence=color_palette
            )

            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#553E6B")
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

            # -----------------------------
            # DONUT CHART
            # -----------------------------
            st.write("## 🍩 Confidence Donut Chart")

            fig_donut = px.pie(
                df_scores,
                names="Category",
                values="Confidence",
                hole=0.5,
                color_discrete_sequence=color_palette
            )

            fig_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#553E6B")
            )

            st.plotly_chart(
                fig_donut,
                use_container_width=True
            )

    else:
        st.warning("⚠ Please enter some news text to classify.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption("✨ Powered by Linear SVM + TF-IDF")  
