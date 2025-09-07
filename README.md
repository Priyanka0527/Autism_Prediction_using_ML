# Autism_Prediction_using_ML
Abstract :
<br> 
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition 
characterized by challenges in social interaction, communication, and repetitive 
behaviors. Early detection of ASD is crucial for timely intervention, which can 
significantly improve the quality of life for affected individuals and their families. 
Traditional diagnostic methods rely heavily on clinical observations and standardized 
testing, which can be time-consuming and subjective.
 This project aims to develop a machine learning-based model to detect autism using 
demographic, behavioral, and historical data. By leveraging a dataset containing 
features such as age, gender, ethnicity, presence of jaundice at birth, prior usage of 
autism screening apps, and familial relations, we aim to create a predictive model that
 can assist in the early detection of ASD. The data is preprocessed to handle missing 
values and categorical variables, followed by feature scaling to standardize the inputs.
Here I implement several machine learning algorithms, including Decision Trees, Support 
Vector Machines, and Random Forests, to identify the most effective model. The 
models are trained and validated using an 80-20 train-test split, and their 
performance is evaluated using metrics such as accuracy, precision, recall, F1-score, 
and the ROC-AUC curve.
Here results indicate that the Random Forest classifier provides the best performance, 
achieving high accuracy and robust predictive capabilities. The feature importance 
analysis reveals key factors that contribute to the prediction of ASD, providing insights
 into the most significant indicators of the disorder.
 The proposed machine learning approach offers a promising supplementary tool for 
early autism detection, potentially enabling healthcare professionals to identify at-risk 
individuals more efficiently and accurately. This project underscores the potential of 
machine learning in enhancing diagnostic processes and highlights the importance of 
integrating technological advancements in medical practice.
<br> 
 Methodology:
 <br> 
 The methodology involves several key steps: data collection, preprocessing, feature 
selection, model training, evaluation, and feature importance analysis. Each step is 
crucial for building an accurate and reliable predictive model.
 1. Data Collection  <br> 
 The dataset used in this project contains various features related to individuals' 
demographics, behaviors, and medical history. The primary columns in the dataset 
include age, gender, ethnicity, presence of jaundice at birth, previous use of autism 
screening apps, and familial relationships. This diverse set of features provides a 
comprehensive basis for training a machine learning model to detect ASD.
 2. Data Preprocessing <br> 
 Data preprocessing is a critical step to ensure the dataset is clean and suitable for 
model training. The preprocessing steps include:
 Handling Missing Values: Identify and handle any missing values in the dataset. 
Common strategies include imputation with mean/median values for numerical 
features or the most frequent category for categorical features.
 Encoding Categorical Variables: Convert categorical variables into numerical format 
using techniques like Label Encoding or One-Hot Encoding. Label Encoding is used
 for ordinal categorical variables, while One-Hot Encoding is preferred for nominal 
categorical variables.
 Feature Scaling: Standardize the numerical features to have a mean of zero and a 
standard deviation of one. This ensures that all features contribute equally to the 
model training process.
 3. Feature Selection <br> 
 Feature selection involves identifying the most relevant features for the predictive 
model. This step can enhance model performance by reducing overfitting and 
improving interpretability. Various techniques can be used for feature selection:
 Correlation Analysis: Analyze the correlation between features and the target 
variable (ASD diagnosis) to identify highly correlated features.
 Feature Importance from Models: Use tree-based models like Random Forest to 
estimate the importance of each feature.
 4. Model Training <br> 
 Several machine learning algorithms are implemented and compared to identify the 
best-performing model. The primary algorithms considered in this project include:
 Decision Trees: A simple and interpretable model that recursively partitions the data
 based on feature values.
 Support Vector Machines (SVM): A powerful classifier that finds the optimal 
hyperplane separating the classes in the feature space.
 Random Forests: An ensemble learning method that combines multiple decision 
trees to improve predictive performance and robustness.
 The dataset is split into training and testing sets using an 80-20 split. The training set 
is used to train the models, and the testing set is used for evaluation.

 5. Model Evaluation <br> 
 Model performance is evaluated using several metrics to ensure a comprehensive 
assessment:
 Confusion Matrix: Provides a summary of the model’s performance by displaying the
 true positives, true negatives, false positives, and false negatives.
 Classification Report: Includes precision, recall, F1-score, and support for each 
class.
 Accuracy Score: Measures the overall accuracy of the model.
 ROC-AUC Curve: Evaluates the model's ability to discriminate between the positive 
and negative classes.
 6. Feature Importance Analysis <br> 
 Analyzing feature importance helps in understanding which features contribute most 
to the model's predictions. This step involves:
 Extracting Feature Importances: Using the trained Random Forest model to extract 
the importance scores of each feature.
 Visualization: Plotting the feature importances to visually assess the contribution of 
each feature


