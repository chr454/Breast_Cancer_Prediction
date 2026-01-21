import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib



#Load the dataset 
print("Loading dataset ...")
data = load_breast_cancer()

X_full = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.Series(data.target)

selected_features = [
  'mean radius',
  'mean texture',
  'mean perimeter',
  'mean area',
  'mean smoothness'
]

X = X_full[selected_features]

#Note: This is not compulsory and is only used to confirm the data in the dataset.
print(f"Original features: {X_full.shape[1]} (30 numeric features)")
print(f"Selected features: {X.shape[1]}")
print(f"Sample: {X.shape[0]}")


model_pipeline = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('classifier', LogisticRegression(max_iter= 10000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Train the model

print("Training in progress ...")
model_pipeline.fit(X_train,y_train)

# Evaluation

print("Evaluating...")
predictions = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)


print(f"accuracy: {accuracy:.4f}")
print(f"recall score: {recall:.4F}")
print(f"precision: {precision:.4f}")

#Save the model

joblib.dump(model_pipeline, 'model/breast_cancer_model.pkl')
joblib.dump(selected_features, 'model_features.pkl')