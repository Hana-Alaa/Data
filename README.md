# <b>Predicting Hazardous Near-Earth Objects (NEOs)</b>
## Project Overview
This project aims to predict whether a Near-Earth Object (NEO) is hazardous or not based on its characteristics. The dataset used contains information on various NEOs and their properties such as size, velocity, and distance from Earth. Machine learning models are employed to classify NEOs into hazardous and non-hazardous categories.

## Approach

### 1. Data Import and Preprocessing:
* The dataset is loaded and cleaned to handle missing values and irrelevant columns.
* Data preprocessing includes encoding categorical variables and scaling numerical features.
* Oversampling techniques (SMOTE) are applied to address class imbalance.
  
### 2. Feature Selection and Engineering:
* The key features used for modeling include the absolute magnitude, estimated diameter, relative velocity, miss distance, and orbiting body.

### 3. Model Training and Tuning:
* Random Forest Classifier was used as the machine learning algorithm.
* The model was tuned using RandomizedSearchCV for hyperparameter optimization.
  
### 4. Evaluation Metrics:
* Accuracy, Precision, Recall, F1 Score, and ROC-AUC were calculated to evaluate model performance.
* Confusion matrices and classification reports were generated for detailed insights.
  

# Random Forest Classifier Model
## Key Findings

### Train Result:
- **Accuracy Score**: 96.88%
- **Classification Report:**

  | Metric       | 0        | 1        | Accuracy | Macro Avg | Weighted Avg |
  |--------------|----------|----------|----------|-----------|--------------|
  | Precision    | 0.979937 | 0.958178 | 0.968796 | 0.969057  | 0.969049     |
  | Recall       | 0.957143 | 0.980432 | 0.968796 | 0.968788  | 0.968796     |
  | F1-Score     | 0.968406 | 0.969177 | 0.968796 | 0.968792  | 0.968792     |
  | Support      | 235855   | 236204   | 0.968796 | 472059    |              |

### Test Result:
- **Accuracy Score**: 94.00%
- **Classification Report:**

  | Metric       | 0        | 1        | Accuracy | Macro Avg | Weighted Avg |
  |--------------|----------|----------|----------|-----------|--------------|
  | Precision    | 0.960654 | 0.920964 | 0.939982 | 0.940809  | 0.940868     |
  | Recall       | 0.917914 | 0.962181 | 0.939982 | 0.940048  | 0.939982     |
  | F1-Score     | 0.938798 | 0.941122 | 0.939982 | 0.939960  | 0.939956     |
  | Support      | 59182    | 58833    | 0.939982 | 118015    |              |

## Model Training and Tuning
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: RandomizedSearchCV was used for hyperparameter optimization.

## Usage
To use this model:
1. Install the required libraries.
2. Load the dataset and preprocess it.
3. Train the model using the provided scripts.
4. Evaluate the model using the provided metrics.

## Requirements
- Python 3.x
- Pandas
- Numpy
- Matplotlib, Seaborn, Plotly
- scikit-learn


