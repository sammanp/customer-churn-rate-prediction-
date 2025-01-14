# Churn Rate Prediction

## Project Overview
This project focuses on predicting customer churn using machine learning techniques. Customer churn refers to the phenomenon where customers stop using a company's products or services. By accurately predicting churn, businesses can take proactive steps to retain customers, optimize marketing efforts, and improve overall customer satisfaction. This project involves leveraging historical customer data and advanced machine learning models to build a robust prediction system.

## Features
- **Data Preprocessing:** Handles missing values, feature encoding, outlier detection, and scaling.
- **Exploratory Data Analysis (EDA):** Identifies patterns, trends, and correlations in the data.
- **Feature Engineering:** Creates meaningful features such as churn rate trends and customer lifetime value.
- **Model Training:** Trains machine learning models like Random Forest, KNN, and SVM.
- **Evaluation:** Assesses models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- **Visualization:** Provides detailed graphs to illustrate data distribution, feature importance, and model performance.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SMOTE (for handling class imbalance)
- XGBoost

## Project Workflow
1. **Data Collection:**
   - Dataset sourced from GitHub: [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d).
   - Includes customer information such as demographics, transaction history, and engagement metrics.

2. **Data Preprocessing:**
   - Handling missing values (e.g., replacing NaN values).
   - Encoding categorical variables using one-hot encoding.
   - Scaling numerical features (e.g., tenure, MonthlyCharges, TotalCharges).
   - Detecting and removing outliers using the Interquartile Range (IQR) method.

3. **Exploratory Data Analysis (EDA):**
   - Correlation analysis to identify relationships between features.
   - Visualization using heatmaps, boxplots, and distribution plots.

4. **Feature Engineering:**
   - Transforming the target variable (Churn: Yes = 1, No = 0).
   - Selecting important features based on domain knowledge and correlation analysis.

5. **Handling Imbalance:**
   - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the target variable.

6. **Model Development:**
   - **Random Forest:** Achieved strong initial performance (Accuracy: 99.86%, F1-score: 0.99862). Tuned parameters for a more generalized model.
   - **KNN:** Demonstrated balanced performance with Accuracy: 81.13%, F1-score: 0.799.
   - **SVM:** Achieved Accuracy: 76%, but exhibited class imbalance requiring further optimization.

7. **Evaluation:**
   - Used metrics such as confusion matrix, precision, recall, F1-score, and ROC-AUC to evaluate model performance.

## Key Files
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, and model development.
- `models/`: Saved trained models.
- `src/`: Source code for preprocessing, training, and evaluation scripts.
- `README.md`: Project documentation.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/churn-rate-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd churn-rate-prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Preprocess the data:
   ```bash
   python src/preprocess.py
   ```
5. Train the models:
   ```bash
   python src/train_model.py
   ```
6. Evaluate the models:
   ```bash
   python src/evaluate_model.py
   ```

## Results
### Model Performance
- **Random Forest:**
  - Accuracy: 81.13%
  - Precision: 79.7%
  - Recall: 81.1%
  - F1-score: 79.9%

- **KNN:**
  - Accuracy: 81.13%
  - Precision: 79.7%
  - Recall: 81.1%
  - F1-score: 79.9%

- **SVM:**
  - Accuracy: 76%
  - Performance significantly impacted by class imbalance.

## Future Enhancements
- Incorporate deep learning models like LSTM or CNN for sequential data analysis.
- Implement explainable AI techniques (e.g., SHAP) for better interpretability.
- Develop a user-friendly dashboard for real-time churn insights.
## License
This project is licensed under the MIT License. See `LICENSE` for details.
## Acknowledgments
- IBM Telco Customer Churn Dataset(https://github.com/IBM/telco-customer-churn-on-icp4d)
- Open-source libraries and tools that made this project possible.

