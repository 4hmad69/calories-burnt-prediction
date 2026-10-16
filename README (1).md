# Calories Burnt Prediction

Predicting calories burnt during exercise with linear regression using Python, NumPy, Pandas, and Matplotlib.

The project compares Gradient Descent implemented from scratch with Ordinary Least Squares (OLS). It also checks how adding engineered features affects prediction accuracy.

## Dataset

Two CSV files contain 15,000 records each:

- `exercise.csv`: participant and exercise information
- `calories.csv`: calories burnt

The files are merged using `User_ID`. The original predictors are Gender, Age, Height, Weight, Duration, Heart_Rate, and Body_Temp. Calories is the target; User_ID is excluded from modeling.

The dataset files are not included in this repository. To reproduce the results, provide the original exercise and calories files described above.

## What the project covers

- Missing-value, duplicate, and outlier checks
- Gender encoding and a reproducible 80/20 train/test split
- Scatterplots to examine linearity
- Three engineered features: BMI, Duration × Heart_Rate, and Duration²
- Feature scaling using training statistics only
- Gradient Descent with five learning rates
- OLS using NumPy to solve the normal equations
- Comparison using MAE, RMSE, R², coefficients, and training time

## Learning rate experiment

Gradient Descent was tested with learning rates of `0.001`, `0.01`, `0.1`, `0.5`, and `1.0`, allowing up to 10,000 updates per run.

The learning rate `0.1` achieved the lowest final training MSE among the tested rates. Rates `0.5` and `1.0` diverged and triggered a numerical safeguard.

## Results

All models were evaluated on the same 3,000-record test set.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| GD: raw + engineered features | 5.8368 | 7.8981 | 0.9841 |
| OLS: raw + engineered features | 5.8491 | 7.8996 | 0.9841 |
| OLS: raw features only | 8.4085 | 11.3131 | 0.9673 |

GD and OLS achieved nearly identical test performance. OLS trained in approximately 0.0018 seconds, compared with 1.7623 seconds for the selected GD run. These are single-run Colab measurements and depend on the environment.

Adding the three engineered features reduced OLS test MAE by about 30.4% and RMSE by 30.2%.

## Repository files

| File | Contents |
|---|---|
| `Calories_Prediction.ipynb` | Code, saved outputs, and plots |
| `Calories_Prediction_Report.pdf` | Methods, results, comparisons, and limitations |
| `requirements.txt` | Python libraries used |
| `.gitignore` | Temporary files and dataset files excluded from Git |

## Run in Google Colab

1. Download `Calories_Prediction.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/) and upload the notebook.
3. Run the first code cell and upload `exercise.csv` and `calories.csv` when prompted.
4. Run the remaining cells in order.

The notebook uses Colab's file-upload feature. NumPy, Pandas, and Matplotlib are available in Colab. For a local environment, install the libraries with `pip install -r requirements.txt` and replace the Colab upload cell with local file loading.

## Limitations

Results are based on one train/test split. Some high-calorie records remain underpredicted, and the model can produce negative predictions. GD had not fully reached the OLS solution after 10,000 updates. Correlated predictors also make individual coefficients harder to interpret.
