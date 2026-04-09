# 🫀 Heart Failure Prediction App

## 📌 Overview
This project uses a **RandomForestClassifier** wrapped in a scikit‑learn pipeline to predict the likelihood of heart disease based on patient health data.  
The pipeline handles preprocessing (encoding categorical features, scaling numericals), model training, evaluation, and deployment via **Streamlit**.

You can try the live app here:  
👉 [Heart Failure Prediction App](https://heartfailureprediction-4n.streamlit.app/)

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/heart-failure-prediction.git
cd heart-failure-prediction
pip install -r requirements.txt
```
## 📂 Project Structure
├── HeartFailure.ipynb          **Jupyter notebook to train and save the pipeline**

├── app.py                      **Streamlit app for predictions**

├── heart_disease_pipeline.pkl  **Saved model**

├── requirements.txt            **Dependencies**

└── README.md                   **Documentation**


# 🚀 Usage
1. Train and Save Model

2. Run Streamlit App Locally
 
Launch the interactive app:

```bash
streamlit run app.py
```
Or use the deployed version here:
👉 [App](https://heartfailureprediction-4n.streamlit.app/)

## 📊 Model Evaluation
The pipeline is evaluated using:

- Accuracy

- Classification Report

- Confusion Matrix

- Permutation Importance

- LIME explanations for individual predictions

## 🧠 Interpretability

- Permutation Importance → Global feature ranking

- LIME → Local explanations for individual patients

This helps understand both overall model behavior and case‑by‑case reasoning.

## 🛠️ Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- streamlit
- lime

## 🌐 Demo
After running streamlit run app.py, you’ll see a web interface where you can:

- Enter patient details (Age, Sex, Cholesterol, RestingBP, MaxHR, Oldpeak, ExerciseAngina, FastingBS, ChestPainType, ST_Slope, RestingECG)

- Get predictions (No Disease or Disease)

- View prediction probabilities

- Explore feature importance and explanations
