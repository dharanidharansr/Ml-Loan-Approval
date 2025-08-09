# 🏦 LoanGenius - Smart Loan Approval Predictor

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-3.0+-green.svg" alt="Flask Version">
  <img src="https://img.shields.io/badge/scikit--learn-1.7+-orange.svg" alt="Scikit-learn Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

<div align="center">
  <h3>🚀 An intelligent machine learning web application that predicts loan approval decisions in real-time</h3>
</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🎯 Demo](#-demo)
- [🛠️ Technologies Used](#️-technologies-used)
- [📊 Dataset](#-dataset)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [🔮 Model Performance](#-model-performance)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

🎨 **Beautiful Modern UI**
- Responsive gradient design with glassmorphism effects
- Interactive form validation and user feedback
- Mobile-friendly interface with smooth animations

🧠 **Advanced Machine Learning**
- Random Forest classifier with GridSearchCV optimization
- Real-time prediction capabilities
- High accuracy loan approval predictions

⚡ **Smart Processing**
- Intelligent input validation and preprocessing
- Error handling with user-friendly messages
- Support for various input formats (with/without commas)

🔒 **Robust Architecture**
- Flask-based web application
- RESTful API endpoints
- Production-ready error handling

---

## 🎯 Demo

### Web Interface
The application features a sleek, modern interface where users can input their financial information:

**Input Fields:**
- 👨‍🎓 Education Level (Graduate/Not Graduate)
- 💼 Employment Type (Self-Employed/Salaried)
- 💰 Annual Income
- 🏠 Loan Amount
- ⏱️ Loan Term (Years)
- 📊 CIBIL Score
- 🏘️ Residential Assets Value
- 🏢 Commercial Assets Value
- 💎 Luxury Assets Value
- 🏛️ Bank Assets Value

**Sample Prediction:**
```
✅ APPROVED: Your loan application has excellent approval chances!
```

---

## 🛠️ Technologies Used

### Backend
- **Python 3.11+** - Core programming language
- **Flask 3.0+** - Web framework
- **scikit-learn 1.7+** - Machine learning library
- **pandas 2.3+** - Data manipulation and analysis
- **numpy 2.3+** - Numerical computing
- **pickle** - Model serialization

### Frontend
- **HTML5** - Markup language
- **CSS3** - Styling with modern gradients and animations
- **JavaScript** - Client-side interactivity
- **Google Fonts (Inter)** - Typography

### Machine Learning
- **Random Forest Classifier** - Primary algorithm
- **GridSearchCV** - Hyperparameter optimization
- **Feature Engineering** - Data preprocessing and transformation

---

## 📊 Dataset

The model is trained on a comprehensive loan approval dataset with **4,269 records** containing:

| Feature | Description | Type |
|---------|-------------|------|
| Education | Graduate (1) / Not Graduate (0) | Binary |
| Self Employed | Self-Employed (1) / Salaried (0) | Binary |
| Income Annual | Yearly income in rupees | Numerical |
| Loan Amount | Requested loan amount | Numerical |
| Loan Term | Loan duration in years | Numerical |
| CIBIL Score | Credit score (300-900) | Numerical |
| Residential Assets | Value of residential properties | Numerical |
| Commercial Assets | Value of commercial properties | Numerical |
| Luxury Assets | Value of luxury items | Numerical |
| Bank Assets | Bank account balance and deposits | Numerical |

**Target Variable:** Loan Status (Approved/Rejected)

---

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dharanidharansr/Ml-Loan-Approval.git
   cd Ml-Loan-Approval
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv ml
   ml\Scripts\activate

   # macOS/Linux
   python -m venv ml
   source ml/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python test_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 💻 Usage

### Web Interface

1. **Open the application** in your web browser
2. **Fill in all required fields** with your financial information
3. **Click "GET MY LOAN PREDICTION"** to submit your application
4. **View the prediction result** with detailed feedback

### API Usage

You can also use the prediction endpoint directly:

```python
import requests

# Prepare your data
data = {
    'feature1': 1,      # Education (1=Graduate, 0=Not Graduate)
    'feature2': 0,      # Self-Employed (1=Yes, 0=No)
    'feature3': 500000, # Annual Income
    'feature4': 1000000,# Loan Amount
    'feature5': 15,     # Loan Term (Years)
    'feature6': 750,    # CIBIL Score
    'feature7': 5000000,# Residential Assets
    'feature8': 2000000,# Commercial Assets
    'feature9': 1000000,# Luxury Assets
    'feature10': 3000000# Bank Assets
}

# Make prediction request
response = requests.post('http://127.0.0.1:5000/predict', data=data)
print(response.text)
```

### Input Guidelines

- **CIBIL Score:** Enter a value between 300-900
- **Monetary Values:** Can include commas (e.g., 5,00,000 or 500000)
- **Education:** 1 for Graduate, 0 for Not Graduate
- **Employment:** 1 for Self-Employed, 0 for Salaried
- **Loan Term:** Enter in years (typically 1-30)

---

## 🔮 Model Performance

The machine learning model achieves excellent performance metrics:

- **Algorithm:** Random Forest Classifier with GridSearchCV
- **Training Data:** 4,269 loan applications
- **Features:** 10 financial and demographic variables
- **Optimization:** Hyperparameter tuning for optimal performance

### Key Features of the Model:
- ✅ Handles both numerical and categorical data
- ✅ Robust to outliers and missing values
- ✅ Provides probability estimates for predictions
- ✅ Feature importance analysis available

---

## 📁 Project Structure

```
Ml-Loan-Approval/
│
├── 📄 app.py                    # Flask web application
├── 📄 model.pkl                 # Trained ML model
├── 📄 loan_approval_dataset.csv # Training dataset
├── 📄 MLPRJCT.ipynb            # Jupyter notebook with model development
├── 📄 test_model.py            # Model testing script
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # Project documentation
│
├── 📁 templates/
│   └── 📄 index.html          # Main web interface
│
├── 📁 ml/                     # Virtual environment
│   ├── 📁 Scripts/           # Python executables
│   ├── 📁 Lib/               # Installed packages
│   └── 📄 pyvenv.cfg         # Environment configuration
│
└── 📁 __pycache__/           # Python cache files
```

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute:
1. 🐛 **Report Bugs** - Found an issue? Let us know!
2. 💡 **Suggest Features** - Have ideas for improvements?
3. 🔧 **Submit Pull Requests** - Ready to contribute code?
4. 📖 **Improve Documentation** - Help others understand the project better

### Development Setup:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support & Contact

- 📧 **Email:** dharanidharansr@example.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/dharanidharansr/Ml-Loan-Approval/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/dharanidharansr/Ml-Loan-Approval/discussions)

---

<div align="center">
  <h3>🌟 If you found this project helpful, please give it a star! ⭐</h3>
  <p>Built with ❤️ using Python, Flask, and scikit-learn</p>
</div>

---

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/dharanidharansr/Ml-Loan-Approval.git
cd Ml-Loan-Approval
python -m venv ml
ml\Scripts\activate  # Windows
pip install -r requirements.txt

# Run the application
python app.py

# Test the model
python test_model.py
```

**Visit:** http://127.0.0.1:5000 🌐