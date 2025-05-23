# E-commerce Analytics Dashboard

## Overview
This project is an interactive analytics dashboard built with Streamlit that provides comprehensive analysis of e-commerce data. It includes features for delivery delay predictions, return request analysis, and customer sentiment analysis.

## Features
- 📊 Interactive data visualization
- 🤖 Machine learning predictions for delivery delays and returns
- 📈 Sentiment analysis of customer reviews
- 📱 Responsive dashboard interface
- 🔄 Real-time data processing
- 📉 KPI monitoring by platform
- 📊 Correlation analysis
- 📦 Product category analysis

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Git

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Activate your virtual environment (if not already activated):
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Project Structure
- `streamlit_app.py`: Main application file
- `requirements.txt`: Project dependencies
- `datos_limpios.csv`: Cleaned dataset
- `cached_predictions.csv`: Cached prediction results
- `Documentacion.txt`: Detailed project documentation

## Key Features

### 1. Data Analysis
- Temporal analysis of orders
- Platform-specific KPIs
- Correlation analysis
- Category and return analysis
- Sentiment analysis of customer reviews

### 2. Predictive Analytics
- Delivery delay prediction
- Return probability prediction
- Real-time prediction interface

### 3. Visualization
- Interactive charts and graphs
- Platform-specific metrics
- Distribution analysis
- Sentiment distribution

## Technical Details
- Built with Streamlit for interactive web interface
- Uses BERT for sentiment analysis
- Implements caching for performance optimization
- Integrates with Kaggle for data access
- Utilizes scikit-learn for machine learning models

## Contributing
Feel free to submit issues and enhancement requests!

## License
[Your License Here]

## Contact
[Your Contact Information] 