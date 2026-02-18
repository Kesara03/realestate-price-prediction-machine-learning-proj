# Property Price Prediction Tool

Full-stack machine learning application for real-time property price estimation. End-to-end pipeline from raw data processing to interactive web deployment.

## Architecture

```
Kaggle Dataset --> Data Cleaning --> Feature Engineering --> Model Training --> Flask API --> Web UI
      |                |                    |                     |               |            |
    CSV           Null handling      Dimensionality          GridSearchCV     REST        HTML/CSS/JS
                  Outlier removal    reduction               Cross-val      endpoints    jQuery AJAX
```

## Data Pipeline

### Dataset
Source: [Bangalore House Price Data (Kaggle)](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

| Feature | Description |
|---------|-------------|
| location | Property location (categorical) |
| total_sqft | Total square footage |
| bath | Number of bathrooms |
| bhk | Number of bedrooms |
| price | Price in Lakhs (target) |

### Data Cleaning
- Null value removal across all features
- Square footage range conversion (e.g., "2100-2850" to average)
- Feature extraction: BHK parsed from size string

### Outlier Removal

**Business Logic Filters:**
- Properties with < 300 sqft per bedroom removed (data errors)
- Properties with bathrooms > bedrooms + 2 removed

**Statistical Filters:**
- Price per sqft outliers removed using mean +/- 1 standard deviation per location
- BHK price anomalies removed (3BHK cheaper than 2BHK for same sqft)

### Dimensionality Reduction
Locations with <= 10 data points consolidated into "other" category, reducing feature space significantly while preserving predictive signal.

### Feature Engineering
- One-hot encoding for location categorical variable
- Price per square foot derived feature for outlier detection
- Dropped auxiliary columns post-processing (size, price_per_sqft)

## Model Development

### Algorithm Comparison
GridSearchCV with 5-fold cross-validation across:

| Algorithm | Hyperparameters Tuned |
|-----------|----------------------|
| Linear Regression | fit_intercept, positive |
| Lasso | alpha, selection |
| Decision Tree | max_depth, criterion |

### Final Model
Linear Regression selected based on cross-validation performance (>80% R-squared across all folds).

```python
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)
```

### Model Artifacts
- `Home_prices_model.pickle`: Serialized trained model
- `columns.json`: Feature column names for inference

## API Endpoints

### Flask Backend

**GET /get_location_names**
Returns available locations for dropdown population.

**POST /predict_home_price**
```json
Request:
{
  "total_sqft": 1000,
  "bhk": 2,
  "bath": 2,
  "location": "Indira Nagar"
}

Response:
{
  "estimated_price": 83.45
}
```

## Frontend

Interactive web interface built with HTML, CSS, and JavaScript.

**Features:**
- Square footage input field
- Radio button selectors for bedrooms (1-5) and bathrooms (1-5)
- Dynamic location dropdown populated from API
- Real-time price estimation with USD conversion
- Responsive design with custom styling

## Project Structure

```
.
├── House_price_data.ipynb    # Model development notebook
├── server/
│   ├── server.py             # Flask API
│   ├── util.py               # Helper functions
│   ├── Home_prices_model.pickle
│   └── columns.json
├── client/
│   ├── app.html              # Frontend markup
│   ├── app.css               # Styling
│   └── app.js                # API integration
└── README.md
```

## Running the Application

### Backend
```bash
cd server
pip install flask numpy pandas scikit-learn
python server.py
```

### Frontend
Open `app.html` in browser or serve via HTTP server.

## Dependencies

| Component | Package |
|-----------|---------|
| Data Processing | pandas, numpy |
| Visualization | matplotlib |
| Machine Learning | scikit-learn |
| API | Flask |
| Frontend | jQuery |

## Author

Kesara Rathnasiri
