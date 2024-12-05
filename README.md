# Loan Default Risk Prediction

This project aims to predict the risk of loan defaults using a dataset and machine learning models. The dataset includes various features related to credit history and loan applications, and the models used for prediction include Random Forest, XGBoost, and Decision Tree classifiers. Among these models, XGBoost showed the best performance, and the model was further tuned for better accuracy.

## Project Setup

To set up this project, you will need Python 3.x and the following libraries. You can install the required libraries using the `requirements.txt` file.

### Install Requirements

You can install all the dependencies with the following command:

```bash
pip install -r requirements.txt
```

### Dependencies

- **numpy**: For numerical computations.
- **pandas**: For data manipulation.
- **matplotlib**: For data visualization.
- **scikit-learn**: For machine learning models and evaluation.
- **xgboost**: For the XGBoost classifier.
- **statsmodels**: For statistical models, including Variance Inflation Factor (VIF).
- **scipy**: For statistical tests like Chi-square and ANOVA.

## Files

- `loan_default_risk_prediction.py`: Main script that loads the dataset, processes it, trains the models, and evaluates them.
- `CRM_v1.sav`: Final model saved after hyperparameter tuning with XGBoost.

## Dataset

The dataset contains features such as:

- Credit history metrics (e.g., `Age_Oldest_TL`, `Tot_TL_closed_L12M`, `num_deliq_6_12mts`)
- Personal details (e.g., `MARITALSTATUS`, `EDUCATION`, `GENDER`)
- Loan enquiry and approval details (e.g., `last_prod_enq2`, `first_prod_enq2`, `Approved_Flag`).

The target variable is `Approved_Flag`, which indicates whether the loan was approved or not.

### Case Study 1 Table (Internal Features)
| Column Name               | Description                                        |
|---------------------------|----------------------------------------------------|
| Total_TL                   | Total trade lines/accounts in Bureau               |
| Tot_Closed_TL              | Total closed trade lines/accounts                  |
| Tot_Active_TL              | Total active accounts                              |
| Total_TL_opened_L6M        | Total accounts opened in last 6 Months             |
| Tot_TL_closed_L6M          | Total accounts closed in last 6 months             |
| pct_tl_open_L6M            | Percent accounts opened in last 6 months           |
| pct_tl_closed_L6M          | Percent accounts closed in last 6 months           |
| pct_active_tl              | Percent active accounts                            |
| pct_closed_tl              | Percent closed accounts                            |
| Total_TL_opened_L12M       | Total accounts opened in last 12 Months            |
| Tot_TL_closed_L12M         | Total accounts closed in last 12 months            |
| pct_tl_open_L12M           | Percent accounts opened in last 12 months          |
| pct_tl_closed_L12M         | Percent accounts closed in last 12 months          |
| Tot_Missed_Pmnt            | Total missed Payments                              |
| Auto_TL                    | Count Automobile accounts                          |
| CC_TL                      | Count of Credit card accounts                      |
| Consumer_TL                | Count of Consumer goods accounts                   |
| Gold_TL                    | Count of Gold loan accounts                        |
| Home_TL                    | Count of Housing loan accounts                     |
| PL_TL                      | Count of Personal loan accounts                    |
| Secured_TL                 | Count of secured accounts                          |
| Unsecured_TL               | Count of unsecured accounts                        |
| Other_TL                   | Count of other accounts                            |
| Age_Oldest_TL              | Age of oldest opened account                       |
| Age_Newest_TL              | Age of newest opened account                       |

### Case Study 2 Table (External Features)
| Column Name                       | Description                                         |
|-----------------------------------|-----------------------------------------------------|
| time_since_recent_payment         | Time since recent Payment made                      |
| time_since_first_deliquency       | Time since first Delinquency (missed payment)       |
| time_since_recent_deliquency     | Time since recent Delinquency                       |
| num_times_delinquent              | Number of times delinquent                          |
| max_delinquency_level             | Maximum delinquency level                           |
| max_recent_level_of_deliq         | Maximum recent level of delinquency                 |
| num_deliq_6mts                    | Number of times delinquent in last 6 months         |
| num_deliq_12mts                   | Number of times delinquent in last 12 months        |
| num_deliq_6_12mts                 | Number of times delinquent between last 6 months and last 12 months |
| max_deliq_6mts                    | Maximum delinquency level in last 6 months          |
| max_deliq_12mts                   | Maximum delinquency level in last 12 months         |
| num_times_30p_dpd                 | Number of times 30+ dpd                             |
| num_times_60p_dpd                 | Number of times 60+ dpd                             |
| num_std                           | Number of standard Payments                         |
| num_std_6mts                       | Number of standard Payments in last 6 months       |
| num_std_12mts                      | Number of standard Payments in last 12 months      |
| num_sub                           | Number of sub standard payments - not making full payments |
| num_sub_6mts                       | Number of sub standard payments in last 6 months   |
| num_sub_12mts                      | Number of sub standard payments in last 12 months  |
| num_dbt                           | Number of doubtful payments                         |
| num_dbt_6mts                       | Number of doubtful payments in last 6 months       |
| num_dbt_12mts                      | Number of doubtful payments in last 12 months      |
| num_lss                           | Number of loss accounts                             |
| num_lss_6mts                       | Number of loss accounts in last 6 months           |
| num_lss_12mts                      | Number of loss accounts in last 12 months          |
| recent_level_of_deliq             | Recent level of delinquency                         |
| tot_enq                           | Total enquiries                                     |
| CC_enq                            | Credit card enquiries                               |
| CC_enq_L6m                         | Credit card enquiries in last 6 months             |
| CC_enq_L12m                        | Credit card enquiries in last 12 months            |
| PL_enq                            | Personal Loan enquiries                             |
| PL_enq_L6m                         | Personal Loan enquiries in last 6 months           |
| PL_enq_L12m                        | Personal Loan enquiries in last 12 months          |
| time_since_recent_enq             | Time since recent enquiry                           |
| enq_L12m                          | Enquiries in last 12 months                         |
| enq_L6m                           | Enquiries in last 6 months                          |
| enq_L3m                           | Enquiries in last 3 months                          |
| MARITALSTATUS                     | Marital Status                                      |
| EDUCATION                         | Education level                                     |
| AGE                               | Age                                                 |
| GENDER                            | Gender                                              |
| NETMONTHLYINCOME                  | Net Monthly Income                                  |
| Time_With_Curr_Empr               | Time with current Employer                          |
| pct_of_active_TLs_ever            | Percent active accounts ever                        |
| pct_opened_TLs_L6m_of_L12m        | Percent accounts opened in last 6 months to last 12 months |
| pct_currentBal_all_TL             | Percent current balance of all accounts             |
| CC_utilization                    | Credit card utilization                             |
| CC_Flag                           | Credit card Flag                                    |
| PL_utilization                    | Personal Loan utilization                           |
| PL_Flag                           | Personal Loan Flag                                  |
| pct_PL_enq_L6m_of_L12m            | Percent enquiries PL in last 6 months to last 12 months |
| pct_CC_enq_L6m_of_L12m            | Percent enquiries CC in last 6 months to last 12 months |
| pct_PL_enq_L6m_of_ever            | Percent enquiries PL in last 6 months to last 6 months |
| pct_CC_enq_L6m_of_ever            | Percent enquiries CC in last 6 months to last 6 months |
| max_unsec_exposure_inPct         | Maximum unsecured exposure in percent              |
| HL_Flag                           | Housing Loan Flag                                   |
| GL_Flag                           | Gold Loan Flag                                      |
| last_prod_enq2                    | Latest product enquired for                         |
| first_prod_enq2                   | First product enquired for                          |
| Credit_Score                      | Applicant's credit score                            |
| Approved_Flag                     | Priority levels                                     |

## Model Description

The following machine learning models are used:

1. **Random Forest**: An ensemble method that builds multiple decision trees and combines their results.
2. **XGBoost**: A gradient boosting method that performs well with structured data.
3. **Decision Tree**: A simpler model that splits the data based on feature values.

### Random Forest Model

```
Accuracy: 0.76

Class p1:
Precision: 0.837
Recall: 0.704


F1 Score: 0.765

Class p2:
Precision: 0.796
Recall: 0.928
F1 Score: 0.857

Class p3:
Precision: 0.442
Recall: 0.211
F1 Score: 0.286

Class p4:
Precision: 0.718
Recall: 0.727
F1 Score: 0.722
```

The Random Forest classifier achieved an accuracy of **76.3%**. It performed well on classes `p1` and `p2` but had lower performance on `p3`.

### XGBoost Model

```
Accuracy: 0.78

Class p1:
Precision: 0.824
Recall: 0.761
F1 Score: 0.791

Class p2:
Precision: 0.826
Recall: 0.914
F1 Score: 0.867

Class p3:
Precision: 0.476
Recall: 0.309
F1 Score: 0.375

Class p4:
Precision: 0.734
Recall: 0.736
F1 Score: 0.735
```

XGBoost achieved **78% accuracy**, showing an improvement over Random Forest. It performed exceptionally well on `p2`, with an F1 score of **0.867**.

### Decision Tree Model
```
Accuracy: 0.71

Class p1:
Precision: 0.721
Recall: 0.723
F1 Score: 0.722

Class p2:
Precision: 0.810
Recall: 0.823
F1 Score: 0.817

Class p3:
Precision: 0.343
Recall: 0.331
F1 Score: 0.337

Class p4:
Precision: 0.650
Recall: 0.629
F1 Score: 0.639
```

The Decision Tree model had an accuracy of **71%**, which is lower than both Random Forest and XGBoost, but it still performed decently on `p1` and `p2`.

## Hyperparameter Tuning in XGBoost

We performed hyperparameter tuning on the XGBoost model using grid search to find the optimal parameters.

**Best Hyperparameters:**
```
Best Hyperparameters: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}
```

The best hyperparameters for XGBoost were found to be:
- **learning_rate**: 0.2
- **max_depth**: 3
- **n_estimators**: 200

After tuning, the **test accuracy** was:

```
Test Accuracy: 0.78
```

## Saving the Final Model

The final trained model was saved using pickle for future use.

## Conclusion

The XGBoost model performed the best for this loan default prediction task, achieving an accuracy of **78%** after hyperparameter tuning. Random Forest was a close second, and Decision Tree performed the worst among the three models.

Future work can focus on exploring other feature engineering techniques, additional model types, and fine-tuning other hyperparameters.

## How to Run

1. Clone or download the repository.
2. Install the required dependencies using the `requirements.txt`.
3. Run the script to train and evaluate the models.

```bash
python loan_default_risk_prediction.py
```
