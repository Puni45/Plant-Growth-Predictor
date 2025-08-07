## 🌿 GreenML: Machine Learning for Smart Plant Growth

GreenML is a machine learning project that predicts optimal plant growth conditions using environmental and agricultural parameters. By leveraging decision tree algorithms and advanced preprocessing techniques, this model helps in making data-driven decisions in agriculture.

---

## 🚀 Project Overview

Agriculture is highly dependent on environmental factors like soil type, water availability, sunlight, and more. This project applies machine learning to:
- Analyze agricultural conditions
- Predict the best conditions for plant growth
- Assist in smart farming decisions

---

## 📊 Dataset

The dataset includes the following features:
- `Soil_Type` (categorical)
- `Sunlight_Hours` (numerical)
- `Water_Frequency` (categorical)
- `Fertilizer_Type` (categorical)
- `Temperature` (numerical)
- `Humidity` (numerical)

The target variable is a **categorical label** representing growth suitability or plant health level.

---

## 🧠 ML Model Used

- `DecisionTreeClassifier` from scikit-learn
- Extensive **hyperparameter tuning** using `GridSearchCV`
- **OneHotEncoder** for categorical feature transformation
- Model persisted using `joblib`

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas, NumPy
- scikit-learn
- Google Colab
- Matplotlib / Seaborn (for EDA & visualization)

---

## 📈 Model Performance

- **Best Accuracy (CV Score):** ~69.4%
- **Best Parameters:**  
  ```python
  {
    'ccp_alpha': 0.0,
    'class_weight': 'balanced',
    'criterion': 'entropy',
    'max_depth': 15,
    'max_features': 'log2',
    'min_samples_leaf': 3,
    'min_samples_split': 5,
    'splitter': 'best'
  }

## 🧪 How to Use


## Clone the repository:

git clone https://github.com/Puni45/greenml.git
cd greenml



## Install dependencies:

pip install -r requirements.txt



## Run the notebook:

Open GreenML_Model.ipynb in Jupyter or Google Colab.

Load your own data and test:


import joblib
model = joblib.load("greenml_model.joblib")
model.predict(new_data)


## 🔍 Sample Input Format
Preprocessed numerical values (after OneHotEncoding):


[ -1.195134, 0.069096, -1.266947, 0.072776, 1.328052, 0.086029 ]

Make sure your input matches the encoded format before prediction.

## 📦 Files in Repository
GreenML_Model.ipynb – Full model pipeline (EDA, preprocessing, training)

greenml_model.joblib – Saved trained model

data.csv – Input dataset

README.md – Project documentation



## ✍️ Author
Puneeth Kumar B C

GitHub: @Puni45

##📜 License
This project is licensed under the MIT License. See LICENSE file for details.



---

Let me know if:
- You want a badge (Colab, License, etc.)
- You're hosting it with a demo
- You need a `requirements.txt`

I'll generate them accordingly.
