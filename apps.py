#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load Iris dataset dari URL."""
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    return df


@st.cache_data
def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, float]:
    """Train Random Forest model dan return model dengan test accuracy."""
    X = df.drop("species", axis=1)
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy


def main():
    st.set_page_config(
        page_title="Klasifikasi Iris dengan Streamlit",
        page_icon="🌸",
        layout="wide"
    )
    
    st.title("🌸 Aplikasi Klasifikasi Bunga Iris")
    st.markdown("Masukkan nilai fitur bunga iris untuk mendapatkan prediksi species:")

    # Load data dan model
    df = load_data()
    model, accuracy = train_model(df)

    # Tampilkan metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Sample", len(df))
    with col2:
        st.metric("Akurasi Model", f"{accuracy*100:.2f}%")

    # Input features
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(df['sepal_length'].min()),
            max_value=float(df['sepal_length'].max()),
            value=float(df['sepal_length'].mean())
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(df['sepal_width'].min()),
            max_value=float(df['sepal_width'].max()),
            value=float(df['sepal_width'].mean())
        )
    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=float(df['petal_length'].min()),
            max_value=float(df['petal_length'].max()),
            value=float(df['petal_length'].mean())
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=float(df['petal_width'].min()),
            max_value=float(df['petal_width'].max()),
            value=float(df['petal_width'].mean())
        )

    # Prediksi
    if st.button("🔮 Prediksi Species Iris", type="primary"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        st.success(f"🌼 **Prediksi: {prediction}**")
        
        st.subheader("Probabilitas:")
        probs_df = pd.DataFrame({
            'Species': model.classes_,
            'Probability': probability
        })
        st.dataframe(probs_df.style.format({'Probability': '{:.2%}'}))
        
        # Plot feature importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'feature': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'importance': model.feature_importances_
        })
        st.bar_chart(importance.set_index('feature'))

    # Show dataset
    if st.checkbox("📊 Tampilkan Dataset Iris (150 samples)"):
        st.dataframe(df, use_container_width=True)

    # Instructions
    with st.expander("ℹ️ Cara Menjalankan"):
        st.code("""
streamlit run apps.py
        """, language="bash")


if __name__ == "__main__":
    main()

