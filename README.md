# Student Performance Indicator

A comprehensive machine learning application that predicts student math scores based on various demographic and academic factors. This end-to-end project includes data preprocessing, model training, evaluation, and deployment through a web application.

## Problem Statement

Educational institutions often need to identify students who may require additional academic support. Traditional methods of assessment can be time-consuming and may not capture all relevant factors that influence student performance. This project addresses the challenge of predicting student math scores using multiple demographic and academic variables, enabling educators to:

- Identify students at risk of underperforming
- Allocate resources more effectively
- Implement targeted intervention strategies
- Understand factors that most influence academic performance

The system analyzes student data including gender, ethnicity, parental education level, lunch type, test preparation course completion, and reading/writing scores to predict math performance with high accuracy.

## Technologies and Libraries

### Core Technologies
- **Python 3.10** - Primary programming language
- **Flask** - Web framework for application deployment
- **HTML/CSS** - Frontend development

### Machine Learning Libraries
- **scikit-learn** - Machine learning algorithms and preprocessing
- **XGBoost** - Gradient boosting framework
- **CatBoost** - Categorical boosting algorithm
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **dill** - Serialization library for model persistence

### Data Visualization
- **matplotlib** - Static plotting library
- **seaborn** - Statistical data visualization

### Development and Deployment
- **setuptools** - Package management
- **logging** - Application logging
- **dataclasses** - Data structure management

## Project Structure

```
mlproject/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py  # Feature engineering and preprocessing
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py    # Prediction pipeline
│   │   └── train_pipeline.py      # Training pipeline
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions
├── templates/
│   ├── index.html                 # Homepage template
│   └── home.html                  # Prediction form template
├── static/
│   └── css/
│       └── style.css              # Application styling
├── artifacts/                     # Model artifacts and data
├── notebook/                      # Jupyter notebooks for EDA
├── logs/                          # Application logs
├── app.py                         # Flask application
├── requirements.txt               # Project dependencies
├── setup.py                       # Package configuration
└── README.md                      # Project documentation
```

## Dataset Features

The model uses the following input features to predict math scores:

- **Gender**: Student gender (male/female)
- **Race/Ethnicity**: Ethnic background (Group A-E)
- **Parental Level of Education**: Education level of parents
  - Some high school
  - High school
  - Some college
  - Associate's degree
  - Bachelor's degree
  - Master's degree
- **Lunch Type**: Meal plan type (standard/free or reduced)
- **Test Preparation Course**: Course completion status (none/completed)
- **Reading Score**: Reading assessment score (0-100)
- **Writing Score**: Writing assessment score (0-100)

**Target Variable**: Math Score (0-100)

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/beniamine3155/mlproject.git
cd mlproject
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install project in development mode**
```bash
pip install -e .
```

## Usage

### Training the Model

Run the complete training pipeline:
```bash
python src/components/data_ingestion.py
```

This will:
- Load and preprocess the student dataset
- Split data into training and testing sets
- Train multiple machine learning models
- Evaluate and select the best performing model
- Save the trained model and preprocessor

### Running the Web Application

1. **Start the Flask server**
```bash
python app.py
```

2. **Access the application**
Open your web browser and navigate to:
```
http://localhost:5001
```

3. **Make predictions**
- Fill in the student information form
- Submit to get the predicted math score
- View results on the same page

### API Usage

The application provides a web interface for predictions. To use programmatically:

```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create prediction pipeline
pipeline = PredictPipeline()

# Prepare input data
data = CustomData(
    gender='female',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch='standard',
    test_preparation_course='completed',
    reading_score=85,
    writing_score=82
)

# Get prediction
features = data.get_data_as_data_frame()
prediction = pipeline.predict(features)
print(f"Predicted Math Score: {prediction[0]}")
```

## Model Performance

The system evaluates multiple algorithms and selects the best performer:

- **Random Forest Regressor**
- **Decision Tree Regressor**  
- **Gradient Boosting Regressor**
- **Linear Regression**
- **XGBoost Regressor**
- **CatBoost Regressor**
- **AdaBoost Regressor**

Model selection is based on R-squared score using cross-validation and hyperparameter tuning.

## Features

### Core Functionality
- **Automated Data Pipeline**: Complete ETL process for student data
- **Multiple Algorithm Support**: Comparison of 7 different ML algorithms
- **Hyperparameter Optimization**: Grid search for optimal parameters
- **Model Persistence**: Serialized models for production deployment
- **Real-time Predictions**: Web-based interface for instant predictions

### Web Application Features
- **Responsive Design**: Mobile-friendly user interface
- **Form Validation**: Client-side and server-side validation
- **Loading Animations**: User feedback during processing
- **Error Handling**: Comprehensive error management
- **Logging**: Detailed application and error logging

## Development

### Project Architecture

The project follows a modular architecture with clear separation of concerns:

- **Data Layer**: Handles data ingestion and transformation
- **Model Layer**: Manages model training and evaluation
- **Pipeline Layer**: Orchestrates prediction workflows
- **Application Layer**: Web interface and API endpoints
- **Utility Layer**: Common functions and configurations

### Adding New Features

1. **New Models**: Add to `model_trainer.py` in the models dictionary
2. **New Features**: Update `data_transformation.py` for preprocessing
3. **UI Changes**: Modify templates and static files
4. **API Endpoints**: Extend `app.py` with new routes

### Testing

Run the training pipeline to verify functionality:
```bash
python src/components/data_ingestion.py
```

Check logs in the `logs/` directory for debugging information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Author**: Beniamine Nahid  
**Email**: beniamine3155@gmail.com  
**GitHub**: [beniamine3155](https://github.com/beniamine3155)

## Acknowledgments

- Dataset source: Student Performance Dataset
- Machine learning libraries: scikit-learn, XGBoost, CatBoost
- Web framework: Flask
- Frontend styling: Custom CSS with modern design principles
