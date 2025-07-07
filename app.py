import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import OrdinalEncoder

model = joblib.load('model.pkl') # loading the model
encoder = joblib.load('encoder.pkl') # loading the encoder

model_features = model.feature_names_in_.tolist() # features of the model
model_cats = [feature for feature in model_features if feature in encoder.feature_names_in_] # categorical features of the model

feature_categories = {
    encoder.feature_names_in_[idx]: encoder.categories_[idx].tolist()
    for idx in range(len(encoder.feature_names_in_))
} # categories of the dataframe features

feature_info = {
    feature: feature_categories[feature] if feature in model_cats else 'number'
    for feature in model_features
}


# modified encoder creation

def create_modified_encoder(encoder):
    if not model_cats:
        return None
    
    categories = [feature_info[feature] for feature in model_cats] # categories of the model
    
    
    modified_encoder = OrdinalEncoder(categories=categories) # defining the modified encoder
    dummy_values = [[categories[i][0] for i in range(len(categories))]] # dummy values creation
    modified_encoder.fit(dummy_values) # training the modified encoder
    modified_encoder.feature_names_in_ = np.array(model_cats) # assigining the model categories to the modified encoder
    
    return modified_encoder

modified_encoder = create_modified_encoder(encoder) # creation of the modified encoder



app = Flask(__name__) # defining the flask
@app.route('/')
def home():
    return render_template('index.html', features=feature_info)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict() # taking user input as dictionary
    df = pd.DataFrame([data]) # converting the inputs in a dataframe
    if modified_encoder != None:
        df[model_cats] = modified_encoder.fit_transform(df[model_cats]) # encoding the categorical inputs
    
    predictions = model.predict(df)[0] # prediction using the user inputs
    
    return render_template('index.html', features=feature_info, predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)