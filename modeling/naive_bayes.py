import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """Custom Transformer for MultiLabel Encoding"""
    def fit(self, X, y=None):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(X)
        return self

    def transform(self, X):
        return self.mlb.transform(X)

df = pd.read_csv("clean_results.csv")
fix_na = ["serving_setting", "associated_people", "related_movie", "paired_drink", "hot_sauce_level"]
df[fix_na] = df[fix_na].fillna("none")

df = df.dropna()

# Fix column formarts
df['serving_setting'] = df['serving_setting'].apply(lambda x: x.split(','))
df['associated_people'] = df['associated_people'].apply(lambda x: x.split(','))

# Splitting Data
X = df.drop(columns=['Label'])
y = df['Label']

# Fix multi categorical data
multi_categorical_features = ['serving_setting', 'associated_people']
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(df[multi_categorical_features]), columns=mlb.classes_)

df = df.drop(columns=[multi_categorical_features]).join(skills_encoded)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=1)

# Preprocessing: Handling Numerical & Categorical Data
numerical_features = ['food_complexity', 'num_ingredients', 'expected_cost']
categorical_features = ['related_movie', 'paired_drink', "hot_sauce_level"]

num_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
cat_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))

preprocessor = ColumnTransformer([
    tuple(['num', num_transformer, numerical_features]),
    tuple(['cat', cat_transformer, categorical_features]),
    tuple(['multi_cat', MultiLabelEncoder(), multi_categorical_features])
])

# Naïve Bayes Classifier
model = make_pipeline(preprocessor, GaussianNB())

# Train & Evaluate
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# Accuracy
print(f"Model Accuracy on Training Data: {accuracy_score(y_train, y_pred_train):.2f}")
print(f"Model Accuracy on Validation Data: {accuracy_score(y_val, y_pred_val):.2f}")
