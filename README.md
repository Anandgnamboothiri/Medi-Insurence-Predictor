MediCost Predictor

      A Flask-based machine learning web application that predicts medical insurance charges based on user details such as age, BMI, smoking status, and region. The app uses a Linear Regression model trained on real-world insurance data.

Features

    Web Interface: Enter details through a simple form and get instant predictions.
    
    REST API: Send JSON input and receive insurance cost predictions in real time.
    
    Auto Model Training: If no trained model exists, the app trains one using the dataset.
    
    Lightweight & Easy to Run: Powered by Flask and scikit-learn.

PROJECT STRUCTURE

    ├── model.py               # Main Flask app
    ├── insurance_new.csv      # Dataset used for training
    ├── insurance_model.pkl    # Saved trained model (auto-generated)
    ├── templates/
    │   └── index.html         # Web form interface
    ├── requirements.txt       # Python dependencies
    └── README.md              # Project documentation

Create an virtual Enviornment for efficiancy and reducing the storage problem on our system run the below line in the cmd or in the powershell
      python -m venv venv
      source venv/bin/activate   # On Windows: venv\Scripts\activate

Install dependencies:
      pip install -r requirements.txt
      
Usage
  After installing the dependencies run python model.py
  After go to the local hosting link which you can see on the output terminal. Follow the link

Example Inputs

    Age: Integer (e.g., 35)
    
    Sex: "male" or "female"
    
    BMI: Float (e.g., 27.5)
    
    Children: Integer (e.g., 2)
    
    Smoker: "yes" or "no"
    
    Region: "southwest", "southeast", "northwest", "northeast"

Tech Stack

    Python
    
    Flask (for the web server & API)
    
    Pandas & NumPy (for data handling)
    
    Scikit-learn (for Linear Regression model)

Future Improvements

    Add support for more ML models (e.g., RandomForest, XGBoost).
    
    Deploy on cloud platforms (Heroku, AWS, etc.).
    
    Improve UI/UX with Bootstrap or React frontend.
    
    Add authentication for API access.

This is what i got through this simple project;

WHAT I LEARN FROM IT : 

  Machine Learning Basics

    How to train a Linear Regression model using scikit-learn.
    
    Encoding categorical variables (sex, smoker, region) into numeric form.
    
    Saving and loading trained models with Pickle.
    
  Data Handling
    
    Using pandas to read and preprocess CSV datasets.
    
    Feature selection and preparing input for prediction.
    
  Backend Development with Flask
    
    Creating routes (/, /predict_form, /predict) in Flask.
    
    Handling both HTML form inputs and JSON API requests.
    
    Rendering templates (index.html) with dynamic results.
    
  API Development
    
    Designing a RESTful API endpoint for predictions.
    
    Sending/receiving JSON data and returning predictions in real time.
    
    Full Workflow of an ML Project
    
    Data → Model Training → Saving Model → Loading Model → Serving Predictions.
    
    Understanding how to connect machine learning with web applications.
    
  Software Engineering Practices
    
    Structuring a project with separate templates, dataset, and model files.
    
    Using a requirements.txt file for dependencies.
    
    Making a README.md for documentation.

Visual Explaination Of Output:

  <img width="1902" height="1065" alt="Screenshot 2025-08-27 172651" src="https://github.com/user-attachments/assets/a4ce9296-0913-43b2-bde7-d42e26ed9649" />


          
      
    
