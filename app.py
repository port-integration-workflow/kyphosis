from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        kyp = joblib.load("kyp.pkl")
        
        # Get values through input bars
        age = request.form.get("age")
        number = request.form.get("number")
        start = request.form.get("start")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[age, number, start]], columns = ["Age", "Number", "Start"])
        
        # Get prediction
        prediction = kyp.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("frontend.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)