# 🎓 Student Dropout Prediction System

A Flask web application that predicts whether a student will **Dropout**, remain **Enrolled**, or **Graduate** based on academic and demographic features.  
The model is trained using Logistic Regression and deployed on **Render** with Gunicorn.

---

## 🚀 Features
- User-friendly web form with dropdowns and numeric inputs.
- Predicts 3 outcomes: **Dropout**, **Enrolled**, **Graduate**.
- Course dropdown auto-populated from dataset.
- Clean UI with Bootstrap + custom CSS.
- Ready for deployment on **Render**.

---

## 📂 Project Structure
project/
│── app.py # Flask application
│── dataset.csv # Student dataset
│── log_reg_model.pkl # Trained Logistic Regression model
│── scaler.pkl # StandardScaler used in training
│── label_encoders.pkl # LabelEncoders for categorical features
│── templates/
│ └── index.html # Frontend HTML
│── static/
│ └── style.css # Custom CSS
│── Procfile # Gunicorn start command (for Render)


---

## 🧑‍💻 Model Inputs

The model requires the following features:

- **Curricular units 1st sem (approved)** *(int)*
- **Curricular units 2nd sem (approved)** *(int)*
- **Curricular units 1st sem (grade)** *(int/float)*
- **Debtor** *(Yes/No)*
- **Tuition fees up to date** *(Yes/No)*
- **Scholarship holder** *(Yes/No)*
- **Gender** *(Male/Female)*
- **Educational special needs** *(Yes/No)*
- **Course** *(dropdown list)*

---

## 🎨 UI

- **Bootstrap 4** for layout  
- **Custom CSS** (`static/style.css`) for modern look  
- Color-coded predictions:
  - 🔴 **Dropout** (Red)
  - 🔵 **Enrolled** (Blue)
  - 🟢 **Graduate** (Green)

---

## 📊 Example Prediction

**Input:**
- Gender: Male  
- Scholarship: Yes  
- Debtor: No  
- 1st Sem Approved: 6  
- 2nd Sem Approved: 5  
- 1st Sem Grade: 14  
- Course: Informatics  

**Output:**
Dropout


---

## 🛠️ Tech Stack
- **Backend:** Flask, scikit-learn  
- **Frontend:** HTML, Bootstrap, CSS  
- **Deployment:** Render + Gunicorn  
- **ML:** Logistic Regression, StandardScaler, LabelEncoder  

---

## 🛠️ Future Improvements
- Add more contextual counselling suggestions (tips, resources) based on features  
- Better handling of unseen courses (fallback/default)  
- Add input validation constraints (ranges)  
- Possibly add more models and model comparison  

---

## 🌐 Live Demo
👉 Try it here: [AI Dropout Prediction App](https://ai-dropout-prediction-and-counselling-o7ch.onrender.com)
