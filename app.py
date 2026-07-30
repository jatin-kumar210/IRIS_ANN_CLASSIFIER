"""
Iris Species Classifier — Streamlit App
Based on project_ANN.ipynb

Compares a simple Perceptron baseline against an sklearn MLPClassifier ANN
on the classic Iris dataset, with an interactive live-prediction panel.
(Uses scikit-learn instead of TensorFlow/Keras to avoid native-DLL install issues.)

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

np.random.seed(42)

st.set_page_config(page_title="Iris ANN Classifier", page_icon="🌸", layout="wide")

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


@st.cache_data
def load_data():
    df = sns.load_dataset("iris")
    return df


df = load_data()

st.title("🌸 Iris Species Classifier")
st.caption("Perceptron baseline vs. a Keras ANN — interactive version of project_ANN.ipynb")

# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------

st.sidebar.header("⚙️ Settings")

test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.subheader("ANN Architecture")
hidden1 = st.sidebar.slider("Layer 1 neurons", 4, 64, 16, 4)
hidden2 = st.sidebar.slider("Layer 2 neurons", 4, 32, 8, 4)
dropout_rate = st.sidebar.slider("Dropout rate", 0.0, 0.5, 0.0, 0.1)

st.sidebar.subheader("Training")
epochs = st.sidebar.slider("Epochs", 10, 300, 100, 10)
batch_size = st.sidebar.select_slider("Batch size", options=[4, 8, 16, 32], value=8)

train_button = st.sidebar.button("🚀 Train Models", type="primary")

# --------------------------------------------------------------------------------------
# Preprocessing (cached per settings)
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def preprocess(test_size, random_state):
    X = df.drop(columns=["species"])
    y_raw = df["species"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler


@st.cache_resource(show_spinner=False)
def train_perceptron(X_train_scaled, y_train, random_state):
    per = Perceptron(max_iter=100, random_state=random_state)
    per.fit(X_train_scaled, y_train)
    return per


@st.cache_resource(show_spinner=False)
def train_ann(X_train_scaled, y_train, hidden1, hidden2, dropout_rate, epochs, batch_size, random_state):
    # Hold out a validation slice (mirrors validation_split=0.2 from the Keras version)
    # so we can still plot train vs. validation curves.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    model = MLPClassifier(
        hidden_layer_sizes=(hidden1, hidden2),
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )

    # dropout_rate has no direct MLPClassifier equivalent; alpha (L2 regularization)
    # is used instead to control overfitting when the slider is above 0.
    model.alpha = dropout_rate * 0.1 if dropout_rate > 0 else model.alpha

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for _ in range(epochs):
        model.fit(X_tr, y_tr)
        train_acc_hist.append(model.score(X_tr, y_tr))
        val_acc_hist.append(model.score(X_val, y_val))
        train_loss_hist.append(model.loss_)
        val_loss_hist.append(log_loss(y_val, model.predict_proba(X_val), labels=[0, 1, 2]))

    history = {
        "accuracy": train_acc_hist,
        "val_accuracy": val_acc_hist,
        "loss": train_loss_hist,
        "val_loss": val_loss_hist,
    }
    return model, history


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

tab_data, tab_train, tab_predict = st.tabs(
    ["📊 Data Overview", "🧠 Train & Evaluate", "🔮 Live Prediction"]
)

# ---------------- Data Overview ----------------
with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class distribution")
        st.bar_chart(df["species"].value_counts())
    with col2:
        st.subheader("Summary statistics")
        st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Feature relationships")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

# ---------------- Train & Evaluate ----------------
with tab_train:
    if train_button:
        st.cache_resource.clear()

    with st.spinner("Preparing data..."):
        X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler = preprocess(
            test_size, random_state
        )

    col1, col2 = st.columns(2)

    # --- Perceptron ---
    with col1:
        st.subheader("Perceptron baseline")
        per = train_perceptron(X_train_scaled, y_train, random_state)
        y_pred_per = per.predict(X_test_scaled)
        acc_per = accuracy_score(y_test, y_pred_per)
        st.metric("Test Accuracy", f"{acc_per:.2%}")
        with st.expander("Classification report"):
            st.text(classification_report(y_test, y_pred_per, target_names=encoder.classes_))
        with st.expander("Confusion matrix"):
            cm = confusion_matrix(y_test, y_pred_per)
            fig_cm, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig_cm)

    # --- ANN ---
    with col2:
        st.subheader("ANN (MLPClassifier)")
        with st.spinner("Training ANN..."):
            model, history = train_ann(
                X_train_scaled, y_train, hidden1, hidden2, dropout_rate, epochs, batch_size, random_state
            )
        y_pred_ann = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred_ann)
        loss = log_loss(y_test, model.predict_proba(X_test_scaled), labels=[0, 1, 2])
        st.metric("Test Accuracy", f"{acc:.2%}", delta=f"loss {loss:.3f}")

        with st.expander("Classification report"):
            st.text(classification_report(y_test, y_pred_ann, target_names=encoder.classes_))
        with st.expander("Confusion matrix"):
            cm_ann = confusion_matrix(y_test, y_pred_ann)
            fig_cm2, ax2 = plt.subplots()
            sns.heatmap(
                cm_ann, annot=True, fmt="d", cmap="Greens",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax2,
            )
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig_cm2)

    st.subheader("ANN training curves")
    fig_hist, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))
    ax_acc.plot(history["accuracy"], label="Train")
    ax_acc.plot(history["val_accuracy"], label="Validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()

    ax_loss.plot(history["loss"], label="Train")
    ax_loss.plot(history["val_loss"], label="Validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    st.pyplot(fig_hist)

# ---------------- Live Prediction ----------------
with tab_predict:
    st.subheader("Try it yourself")
    st.write("Adjust the measurements below and see what each model predicts.")

    X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler = preprocess(
        test_size, random_state
    )
    per = train_perceptron(X_train_scaled, y_train, random_state)
    model, history = train_ann(
        X_train_scaled, y_train, hidden1, hidden2, dropout_rate, epochs, batch_size, random_state
    )

    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    with c2:
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3, 0.1)

    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )
    input_scaled = scaler.transform(input_df)

    pred_per = encoder.inverse_transform(per.predict(input_scaled))[0]
    probs_ann = model.predict_proba(input_scaled)[0]
    pred_ann = encoder.classes_[np.argmax(probs_ann)]

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Perceptron prediction:** {pred_per}")
    with col2:
        st.success(f"**ANN prediction:** {pred_ann}")
        prob_df = pd.DataFrame({"species": encoder.classes_, "probability": probs_ann})
        st.bar_chart(prob_df.set_index("species"))
