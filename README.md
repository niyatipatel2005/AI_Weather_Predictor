# AI Weather Predictor 🌤️

**AI Weather Predictor** is a Python-based machine learning project that predicts weather conditions using historical weather data. This project demonstrates data preprocessing, feature engineering, and predictive modeling with a user-friendly interface.

---

## **Features**
- Predict weather conditions based on input data.
- User-friendly interface with Streamlit.
- Quick and accurate predictions using a trained machine learning model.
- Easy to run locally.

---

## **Tech Stack**
- **Python 3.x**
- **Pandas** – for data manipulation
- **Scikit-learn** – for machine learning
- **Joblib** – for saving/loading the trained model
- **Streamlit** – for building the interactive web app

---

## **Project Structure**

```bash
AI_Weather_Predictor/
│
├── app.py # Streamlit app
├── model.py # Model training script
├── weather_model.pkl # Trained model (not included in repo, for this run the model.py file first)
├── requirements.txt # Python dependencies
├── data/ # Sample datasets
└── README.md
```


> ⚠️ **Note:** `weather_model.pkl` is not included in the repo due to size. You can get it by running the model.py file.

---

## **Installation**
1. Clone the repo:
```bash
git clone https://github.com/niyatipatel2005/AI_Weather_Predictor.git
cd AI_Weather_Predictor
```
2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the environment:
```bash
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Usage

- first run the model.py file by this command :
  ```bash
  python model.py
  ```
  
- Run the Streamlit app:
  ```bash
   streamlit run app.py
  ```

- Enter your input data in the app and get predictions instantly.

## Future Work

- Add real-time weather API integration.
- Enhance prediction accuracy with more datasets.
- Deploy as a web app for global access.


## Authors

* Niyati Patel  -  https://github.com/niyatipatel2005


## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
