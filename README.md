# **Bank Churn Prediction (AUC = 0.99) | Percentage 97.1% Correct**
---
by [Dwi Pamuji Bagaskara](https://github.com/DwiPamuji)

<img src="assets/online-banking-concept-making-financial-operations_277904-2297.jpg" alt="Online banking concept. making financial operations by inspiring - www.freepik.com"/>

*Image Source: https://www.freepik.com/free-vector/online-banking-concept-making-financial-operations_10007235.htm#query=BANK%20CHURN&position=10&from_view=search*

# **Contents**
---
1. [Business Problem Understanding](#business-problem-understanding)
1. [Data Understanding](#data-understanding)
1. [Exploratory Data Analysis](#exploratory-data-analysis)
1. [Data Preprocessing](#data-preprocessing)
1. [Modeling and Evaluation](#modeling-and-evaluation)
1. [Hyperparameter Tuning Best Models (XGBoost)](#hyperparameter-tuning-best-models-(XGBoost))
1. [Compare Model Predict and Actual](#compare-model-predict-and-actual)
1. [Test Machine Learning](#test-machine-learning)
1. [Conclusion](#conclusion)

# <a id="business-problem-understanding">**Business Problem Understanding**</a> 
---
## **Context**

Almost all bank companies have credit card facilities, to keep customers loyal and move to other cradit card a challenge for each bank companies. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

Target :

0 : Existing Customer (Customer will stay)

1 : Attrited Customer (Customer will churn)

## **Problem Statement**

In the field of banking companies, apart from looking for new customers, it is important to keep customers in our company. how the bank retains its customers is to give more attention or more promotion to customers who are about to leave the bank so that customers don't leave the bank

And if attention and promotion are given equally to all customers, then the costs will be large

## **Goals**

So based on these problems, the company wants to have the ability to predict the possibility that someone will churn or move to another company or not, so they can focus the promo on customers who will churn so they don't move to other banks.

and also, the company wants to know what factors/variables make someone churn and switch to another company

## **Analytical Approach**

So what we're going to do is analyze the data to find patterns that differentiate customers who will churn or not. Then we will create a classification model that will help companies predict the probability that a candidate will churn

# <a id="data-understanding">Data Understanding</a>
---

Dataset are obtained from: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

> "A business manager of a consumer credit card portfolio is facing the problem of customer attrition. They want to analyze the data to find out the reason behind this and leverage the same to predict customers who are likely to drop off.." 

## **Attributes Information**

| **Attribute** | **Data Type** | **Description** |
| --- | --- | --- |
| CLIENTNUM | int64 | Client number. Unique identifier for the customer holding the account |
| Attrition_Flag | object | Internal event (customer activity) variable - if the account is closed then 1 else 0 |
| Customer_Age | int64 | Demographic variable - Customer's Age in Years |
| Gender | object | Demographic variable - M=Male, F=Female |
| Dependent_count | object | Demographic variable - Number of dependents |
| Education_Level | object | Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc. |
| Marital_Status | object | Demographic variable - Married, Single, Divorced, Unknown |
| Income_Category | object | Demographic variable - Annual Income Category of the account holder (< 40K, 40K-60K, 60K-80K, 80K-120K, > 120k) |
| Card_Category | object | Product Variable - Type of Card (Blue, Silver, Gold, Platinum |
| Months_on_book | int64 | Period of relationship with bank |
| Total_Relationship_Count | int64 | Total no. of products held by the customer |
| Months_Inactive_12_mon | int64 | No. of months inactive in the last 12 months |
| Contacts_Count_12_mon | int64 | No. of Contacts in the last 12 months |
| Credit_Limit | float64 | Credit Limit on the Credit Card |
| Total_Revolving_Bal | int64 | Total Revolving Balance on the Credit Card |
| Avg_Open_To_Buy | float64 | Open to Buy Credit Line (Average of last 12 months) |
| Total_Amt_Chng_Q4_Q1 | float64 | Change in Transaction Amount (Q4 over Q1) |
| Total_Trans_Amt | int64 | Total Transaction Amount (Last 12 months) |
| Total_Trans_Ct | int64 | Total Transaction Count (Last 12 months) |
| Total_Ct_Chng_Q4_Q1 | float64 | Change in Transaction Count (Q4 over Q1) |
| Avg_Utilization_Ratio | float64 | Average Card Utilization Ratio |

# <a id="exploratory-data-analysis">Exploratory Data Analysis</a>
---

<img src="assets/EDA 1.jpg" alt="Customer Existing and Customer Attrated"/>

<img src="assets/EDA 2.jpg" alt="Customer Existing and Customer Attrated by Gender"/>

<img src="assets/EDA 3.jpg" alt="Credit Card Type"/>

<img src="assets/EDA 4.jpg" alt="Credit Card Type by Gender"/>

<img src="assets/EDA 5.jpg" alt="Presentage Used Credit By Education"/>

<img src="assets/EDA 6.jpg" alt="Presentage Used Credit by Status"/>

<img src="assets/EDA 7.jpg" alt="Presentage Used Credit by Income"/>

# <a id="data-preprocessing">Data Preprocessing</a>
---

## **Identify Outlier, Duplicates, missing value etc**

One of the most important preprocessing steps in a Data Science project. Some of the things we do in this project are as follows:
- Identify outlier, anomaly, duplicates, and missing value.
- Handling outlier, anomaly, duplicates, and missing value

## **Check Distribution**
<img src="assets/Data Preprocessing 1 - Distribution Plot.jpg" alt="Distribution Plot"/>

## **Check Outlier with Box Plot**
<img src="assets/Data Preprocessing 2 - Box Plot Check Outlier.jpg" alt="Box Plot Check Outlier"/>

Total outlier = 3326 or Percentage Outlier = 32.84%
we will keep outlier because we asume there is social inequality

## **Check Missing Value**
<img src="assets/Data Preprocessing 3 - Check Missing Value.jpg" alt="Check Missing Value"/>

in this case not have missing value

# <a id="Modeling and Evaluation">Modeling and Evaluation</a>
---
## **Confusion Metric**

**Confusion Matrix** is a performance measurement for the machine learning classification problems where the output can be two or more classes. It is a table with combinations of predicted and actual values. A confusion matrix is defined as thetable that is often used to describe the performance of a classification model on a set of the test data for which the true values are known.
<img src="assets/Modeling and Eval 1 - Confution Metric.jpg" alt="Confution Metric"/>
For this case target is :

0 : Existing Customer

1 : Attrited Customer

---
True Positive: We predicted positive and it’s true. **We predicted that customer will churn** and actually **churn**.
True Negative: We predicted negative and it’s true. **We predicted that customer is not churn** and actually **not churn**.
False Positive (Type 1 Error)- We predicted positive and it’s false. **We predicted that customer will churn** and actually **not churn**.
False Negative (Type 2 Error)- We predicted negative and it’s false. **We predicted that customer is not churn** and actually **churn**

for this case we will use metric **F1-Score** cause we will fokus to recall and precision, and to strengthen analysis we will use **ROC/AUC** too, cause this data is imbalace that has large effect on PR but not ROC/AUC.

## **Approaching Categorical Features**

We will separate ordinal variable and nominal variable: 

Ordinal Variable in this data:
1. Income_Category
1. Card_Category
1. Education_Level

Nominal Variable in this data:
1. Gender
1. Dependent_count
1. Attrition_Flag
1. Marital_Status

## **Train Test Split**

Processing Scheme :

1. Target : Attrition_Flag
1. Standard Scaller : Customer_Age, Months_on_book, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1, Total_Trans_Ct
1. Robust Scaller : Credit_Limit, Avg_Open_To_Buy, Total_Trans_Amt, Avg_Utilization_Ratio
1. Out : CLIENTNUM

## **Modeling**

We will compare 7 Model :
1. **Logistic Regression**
2. **K Nearest Neighbors Classifier**
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
5. **AdaBoost Classifier**
6. **Gradient Boosting Classifier**
7. **XGBoost Classifier**

<img src="assets/Modeling and Eval 2 - Model Benchmark.jpg" alt="Model Benchmark"/>

Of the 7 models tested, the 3 best models is :

|No | Model | f1 score mean | roc/auc Mean |
| --- | --- | --- | --- |
|1. | **XGBoost Classifier** | 0.898873 | 0.990857 |
|2. | **Gradient Boost Classifier** | 0.887280	 | 0.987409 |
|3. | **Random Forest Classifier** | 0.858483 | 0.985179 |

# <a id="hyperparameter-tuning-best-models-(XGBoost)">Hyperparameter Tuning Best Models (XGBoost)</a> 
---

**XGBoost is eXtream Gradient Boosting is a specific implementation of the Gradient Boosting Model which uses more accurate approximations to find the best tree model**. XGBoost specifically used a more regularized model formalization to control overfitting, which gives it better perfomance.

**How XGBoost Works?**

<img src="assets/Hyperparameter 1 - XGBoost Work.jpg" alt="XGBoost Works"/>
Source : https://algotech.netlify.app/blog/xgboost/

**XGBoost After Tuning**
Best_score: 0.9901088935933082
Best_params: {'model__subsample': 0.4, 'model__reg_alpha': 0.1668100537200059, 'model__n_estimators': 214, 'model__max_depth': 24, 'model__learning_rate': 0.18, 'model__gamma': 3, 'model__colsample_bytree': 1.0}

**Compare XGBoost Before Tuning and After Tuning**
<img src="assets/Hyperparameter 2 - Compare Before and After Tuning.jpg" alt="XGBoost Works"/>

**XGBoost After Hyperparameter Tuning have lower than XGBoost before tuning, for next we will use XGBoost Before Tuning | XGBClassifier(eval_metric='auc', random_state = 2022)**

## **Best Model**
> XGBClassifier(eval_metric='auc', random_state = 2022)
**ROC AUC Curve**

<img src="assets/Best Model 1 - ROC AUC Curve.jpg" alt="ROC AUC Curve"/>

# <a id="compare-model-predict-and-actual">**Compare XGBoost Predict and Actual**</a> 
---

<img src="assets/Best Model 2 - Compare XGBoost predict with Actual.jpg" alt="Compare xgb predict with actual"/>

<font size = "5"> **Percentage Machine Learning Predict Churn :
Correct 97.1%
Wrong 2.9%**</font>

# <a id="conclusion">**Conclusion**</a> 
---

<font size="10">With this machine learning, bank companies can **predict which customers will stay and will churn**, with **the percentage of wrong is only 2.9% and correct is 97.1%**.</font>

<font size="10">**By knowing which customers will churn, we can provide promotions so that customers do not churn or move to another bank**</font>
