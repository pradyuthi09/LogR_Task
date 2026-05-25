import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

st.set_page_config(
    page_title="Logistic Regression Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Logistic Regression using Titanic Dataset")

st.markdown("""
This app demonstrates:
- Logistic Regression
- Data preprocessing
- Model evaluation
- Visualization
- Prediction system
""")

# Load Dataset
df = pd.read_csv("train.csv")

st.header("🗂 Dataset Preview")
st.dataframe(df.head())

# Data Cleaning
columns = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked',
    'Survived'
]

new_df = df[columns].copy()

new_df['Age'] = new_df['Age'].fillna(new_df['Age'].mean())
new_df['Embarked'] = new_df['Embarked'].fillna(new_df['Embarked'].mode()[0])

new_df['Sex'] = new_df['Sex'].map({'male': 0, 'female': 1})

new_df = pd.get_dummies(new_df, columns=['Embarked'], drop_first=True)

st.header("🧹 Cleaned Dataset")
st.dataframe(new_df.head())

# Split Data
X = new_df.drop('Survived', axis=1)
y = new_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
st.header("🤖 Model Training")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.success("Model Trained Successfully!")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Accuracy
st.header("📈 Model Performance")

accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy Score", f"{accuracy:.2f}")

# Classification Report
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# ROC Curve
st.subheader("📉 ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

fig2, ax2 = plt.subplots(figsize=(6, 5))

ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax2.plot([0, 1], [0, 1], linestyle='--')

ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()

st.pyplot(fig2)

# Feature Importance
st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

importance = importance.sort_values(by='Coefficient', ascending=False)

st.dataframe(importance)

# User Prediction
st.header("🧠 Predict Survival")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])

with col2:
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.slider("Siblings/Spouse", 0, 8, 0)

with col3:
    parch = st.slider("Parents/Children", 0, 6, 0)
    fare = st.slider("Fare", 0, 600, 50)

embarked = st.selectbox("Embarked", ["Q", "S", "C"])

sex_val = 0 if sex == "Male" else 1

embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = np.array([[
    pclass,
    sex_val,
    age,
    sibsp,
    parch,
    fare,
    embarked_Q,
    embarked_S
]])

if st.button("Predict"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Passenger Survived ✅")
    else:
        st.error("Passenger Did Not Survive ❌")

    st.info(f"Survival Probability: {probability:.2f}")


