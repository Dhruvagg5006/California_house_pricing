import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load model & scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        data = [float(x) for x in request.form.values()]

        # Scale input
        final_input = scaler.transform(np.array(data).reshape(1, -1))

        # Predict
        output = regmodel.predict(final_input)[0]

        return render_template(
            "home.html",
            prediction_text=f"Predicted House Price is {output}"
        )

    except Exception as e:
        return render_template(
            "home.html",
            prediction_text=f"Error occurred: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)
