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
1. [Hyperparameter Tuning](#hyperparameter-tuning)
1. [Compare Model Predict VS Actual](#compare-model-predict-vs-actual)
1. [Test Machine Learning](#test-machine-learning)
1. [Conclution](#conclution)

# <a id="business-problem-understanding">Business Problem Understanding</a> 
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













