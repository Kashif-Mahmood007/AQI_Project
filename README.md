# 🌿 AQI Project

This project analyzes Air Quality Index (AQI) data to evaluate environmental conditions and visualize pollution levels using Python-based data science tools.

---

## 🚀 Features
- Data cleaning and preprocessing  
- Visualization of AQI trends  
- Machine Learning model for AQI prediction  
- Supports multiple datasets  

---

## 🧠 Requirements
- Python 3.11  
- All libraries listed in `requirements.txt`

---

## ⚙️ Setup Instructions

1. **Clone the repository**
git clone https://github.com/<your-username>/AQI_Project.git
cd AQI_Project

2. **Create a virtual environment**
py -3.11 -m venv venv
source venv/Scripts/activate  # for Git Bash on Windows

3. **Install dependencies**
pip install -r requirements.txt

4. **Run the Project**
python app.py




For Github Actions,
Create 2 secrets in repo, 
- Name: WAQI_TOKEN,     Value: My AQI API Key (WAQI)
- Name: HOPSWORKS_API_KEY,   Value: My Hopswork API Key 