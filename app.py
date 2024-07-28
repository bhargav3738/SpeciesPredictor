from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Load the model and label encoder
model = pickle.load(open('model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = ""  
    if request.method == 'POST':
        try:
            features = [float(request.form[f'feature{i}']) for i in range(1, 7)]
            # Scale the features
            features_array = scaler.transform([features])
            # Predict the species
            prediction = model.predict(features_array)
            species = label_encoder.inverse_transform(prediction)
            prediction_text = f'Predicted Fish Species: {species[0]}'
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
