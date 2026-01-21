from flask import Flask, render_template, request
import logging
import joblib
import pandas as pd

# Initialize the flask app
app = Flask(__name__)

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

#Load the model and the features
try:
  model = joblib.load('model/breast_cancer_model.pkl')
  feature_names = joblib.load('model/model_features.pkl')
  logger.info("model loaded successfully")
except Exception as e:
  logger.error(f"error loading files: {e}")
  model = None
  feature_names = []


@app.route('/')
def home():
  return render_template('index.html', features = feature_names)

@app.route('/predict', methods=['POST'])
def predict():
  if not model:
    return render_template('index.html', error= "model not loaded", features = feature_names)
  
  try:
    input_data = []
    for feature in feature_names:
      value = float(request.form[feature])
      input_data.append(value)
    

    df = pd.DataFrame([input_data], columns=feature_names)

    prediction = model.predict(df)[0]

    if prediction == 0:
      result = "MALIGNANT (Potential cancer detected)"
      css_class = "danger"
    else:
      result ="BENIGN (Safe)"
      css_class = "success"
    
    return render_template('index.html', prediction=result, css_class=css_class, features = feature_names)
  
  except Exception as e:
    return render_template('index.html', error =f"error: {e}", features = feature_names)
  
if __name__ == "__main__":
  app.run(debug=True)