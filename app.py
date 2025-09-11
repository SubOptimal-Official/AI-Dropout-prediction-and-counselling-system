from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
model = pickle.load(open("log_reg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Load dataset to get dropdown options
data = pd.read_csv("DATA/dataset.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    courses = sorted(data["Course"].dropna().unique())
    prediction = None

    if request.method == "POST":
        # Collect inputs
        sem1_approved = int(request.form.get("sem1_approved"))
        sem2_approved = int(request.form.get("sem2_approved"))
        sem1_grade = int(request.form.get("sem1_grade"))
        debtor = 1 if request.form.get("debtor") == "Yes" else 0
        tuition = 1 if request.form.get("tuition") == "Yes" else 0
        scholarship = 1 if request.form.get("scholarship") == "Yes" else 0
        gender = 1 if request.form.get("gender") == "Male" else 0
        special_needs = 1 if request.form.get("special_needs") == "Yes" else 0
        course = request.form.get("course")

        # Create input DataFrame
        input_df = pd.DataFrame([[
            sem1_approved,
            sem2_approved,
            sem1_grade,
            debtor,
            tuition,
            scholarship,
            gender,
            special_needs,
            course
        ]], columns=[
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (grade)",
            "Debtor",
            "Tuition fees up to date",
            "Scholarship holder",
            "Gender",
            "Educational special needs",
            "Course"
        ])

        # Encode categorical features
        for col, le in label_encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # Scale numerical
        input_scaled = scaler.transform(input_df)

        # Predict
        pred = model.predict(input_scaled)[0]

        # Map prediction
        mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        prediction = mapping.get(pred, "Unknown")

    return render_template("index.html", courses=courses, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
