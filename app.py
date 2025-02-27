from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (Ensure the model exists)
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
except FileNotFoundError:
    print("Error: 'diabetes_model.pkl' not found. Ensure the model exists.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect input values from the form
            features = [float(request.form[field]) for field in [
                "pregnancies", "glucose", "bloodpressure", "skinthickness",
                "insulin", "bmi", "dpf", "age"
            ]]

            # Make prediction
            prediction = model.predict([np.array(features)])[0]

            # Prepare the result message
            if prediction == 1:
                result_message = "Based on the provided information, you may be diabetic. Please consult a medical professional."
            else:
                result_message = "Based on the provided information, you are not diabetic. Maintain a healthy lifestyle!"

            # Redirect to the result page with the prediction
            return redirect(url_for('result', message=result_message))

        except ValueError:
            # Handle invalid input
            return redirect(url_for('result', message="Invalid input. Please enter valid numeric values."))

    return render_template("index.html")


@app.route("/result")
def result():
    # Display the prediction or error message
    message = request.args.get('message', 'Something went wrong. Try again!')
    return render_template("result.html", message=message)


if __name__ == "__main__":
    app.run(debug=True)
