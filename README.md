

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("C:\\Users\\gayat\\Downloads\\garments_worker_productivity (1).csv")
df.head()
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity
0	1/1/2015	Quarter1	sweing	Thursday	8	0.80	26.16	1108.0	7080	98	0.0	0	0	59.0	0.940725
1	1/1/2015	Quarter1	finishing	Thursday	1	0.75	3.94	NaN	960	0	0.0	0	0	8.0	0.886500
2	1/1/2015	Quarter1	sweing	Thursday	11	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570
3	1/1/2015	Quarter1	sweing	Thursday	12	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570
4	1/1/2015	Quarter1	sweing	Thursday	6	0.80	25.90	1170.0	1920	50	0.0	0	0	56.0	0.800382
df['productivity_difference'] = df['actual_productivity'] - df['targeted_productivity']
df.head()
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity	productivity_difference
0	1/1/2015	Quarter1	sweing	Thursday	8	0.80	26.16	1108.0	7080	98	0.0	0	0	59.0	0.940725	0.140725
1	1/1/2015	Quarter1	finishing	Thursday	1	0.75	3.94	NaN	960	0	0.0	0	0	8.0	0.886500	0.136500
2	1/1/2015	Quarter1	sweing	Thursday	11	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
3	1/1/2015	Quarter1	sweing	Thursday	12	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
4	1/1/2015	Quarter1	sweing	Thursday	6	0.80	25.90	1170.0	1920	50	0.0	0	0	56.0	0.800382	0.000382
df['department'] = df['department'].replace('sweing', 'sewing')
df.head()
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity	productivity_difference
0	1/1/2015	Quarter1	sewing	Thursday	8	0.80	26.16	1108.0	7080	98	0.0	0	0	59.0	0.940725	0.140725
1	1/1/2015	Quarter1	finishing	Thursday	1	0.75	3.94	NaN	960	0	0.0	0	0	8.0	0.886500	0.136500
2	1/1/2015	Quarter1	sewing	Thursday	11	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
3	1/1/2015	Quarter1	sewing	Thursday	12	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
4	1/1/2015	Quarter1	sewing	Thursday	6	0.80	25.90	1170.0	1920	50	0.0	0	0	56.0	0.800382	0.000382
df['wip'] = df['wip'].fillna(0)
df.head()
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity	productivity_difference
0	1/1/2015	Quarter1	sewing	Thursday	8	0.80	26.16	1108.0	7080	98	0.0	0	0	59.0	0.940725	0.140725
1	1/1/2015	Quarter1	finishing	Thursday	1	0.75	3.94	0.0	960	0	0.0	0	0	8.0	0.886500	0.136500
2	1/1/2015	Quarter1	sewing	Thursday	11	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
3	1/1/2015	Quarter1	sewing	Thursday	12	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
4	1/1/2015	Quarter1	sewing	Thursday	6	0.80	25.90	1170.0	1920	50	0.0	0	0	56.0	0.800382	0.000382
unique_values = df['department'].unique()
print(unique_values)
['sewing' 'finishing ' 'finishing']
unique_values = df['day'].unique()
print(unique_values)
['Thursday' 'Saturday' 'Sunday' 'Monday' 'Tuesday' 'Wednesday']
df['department'] = df['department'].replace('finishing ', 'finishing')
Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
# Plot histograms for all columns
df.hist(bins=20, figsize=(15, 12))
plt.tight_layout()
plt.show()

numeric_columns = [
    'actual_productivity', 'productivity_difference'
]
sns.pairplot(df[numeric_columns])
plt.show()

x = df['over_time']
y = df['productivity_difference']
plt.scatter(x, y)
plt.xlabel('Overtime (in minutes)')
plt.ylabel('Difference between actual and targeted productivity')
plt.title('Effect of overtime on productivity')
plt.grid(True)
plt.show()

df.groupby('department')['productivity_difference'].mean().plot(kind='bar')
plt.xlabel('Department')
plt.ylabel('Average of productivity difference')
plt.title('Effect of department on average productivity')
plt.show()

max_prod_diff = max(df['productivity_difference'])
min_prod_diff = min(df['productivity_difference'])
print(max_prod_diff)
0.644375
print(min_prod_diff)
-0.561958333
df.groupby('department')['productivity_difference'].median().plot(kind='bar')
plt.xlabel('Department')
plt.ylabel('Median of productivity difference')
plt.title('Effect of department on median productivity')
plt.show()

x = df['incentive']
y = df['productivity_difference']
plt.scatter(x, y)
plt.xlabel('Incentive')
plt.ylabel('Difference between actual and targeted productivity')
plt.title('Effect of incentive on productivity')
plt.grid(True)
plt.show()

plt.scatter(x, y)
plt.xlabel('Incentive')
plt.ylabel('Difference between actual and targeted productivity')
plt.title('Effect of incentive on productivity')
plt.xlim(0, 200)
plt.grid(True)
plt.show()

unique_teams = df['team'].unique()
print(unique_teams)
[ 8  1 11 12  6  7  2  3  9 10  5  4]
Effect of day on productivity
df.groupby('day')['productivity_difference'].mean().plot(kind='bar')
plt.xlabel('Day of the week')
plt.ylabel('Mean of productivity difference')
plt.title('Effect of day on average productivity difference')
plt.show()

plt.scatter(df['targeted_productivity'], df['productivity_difference'])
plt.xlabel('Targeted Productivity')
plt.ylabel('Difference between actual and targeted productivity')
plt.title('Effect of targeted productivity')
plt.grid(True)
plt.show()

Effect of team on productivity
df.groupby('team')['productivity_difference'].median().plot(kind='bar')
plt.xlabel('Team')
plt.ylabel('Mean of productivity difference')
plt.title('Effect of team on average productivity difference')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['department'] = le.fit_transform(df['department'])
df['day'] = le.fit_transform(df['day'])
Prediction Process
Using 4 techniques:

Decision tree
Random forest
SVR
Linear Regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Encoding string columns:
le = LabelEncoder()
df['department'] = le.fit_transform(df['department'])
df['day'] = le.fit_transform(df['day'])
df.head(10)
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity	productivity_difference
0	1/1/2015	Quarter1	1	3	8	0.80	26.16	1108.0	7080	98	0.0	0	0	59.0	0.940725	0.140725
1	1/1/2015	Quarter1	0	3	1	0.75	3.94	0.0	960	0	0.0	0	0	8.0	0.886500	0.136500
2	1/1/2015	Quarter1	1	3	11	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
3	1/1/2015	Quarter1	1	3	12	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
4	1/1/2015	Quarter1	1	3	6	0.80	25.90	1170.0	1920	50	0.0	0	0	56.0	0.800382	0.000382
5	1/1/2015	Quarter1	1	3	7	0.80	25.90	984.0	6720	38	0.0	0	0	56.0	0.800125	0.000125
6	1/1/2015	Quarter1	0	3	2	0.75	3.94	0.0	960	0	0.0	0	0	8.0	0.755167	0.005167
7	1/1/2015	Quarter1	1	3	3	0.75	28.08	795.0	6900	45	0.0	0	0	57.5	0.753683	0.003683
8	1/1/2015	Quarter1	1	3	2	0.75	19.87	733.0	6000	34	0.0	0	0	55.0	0.753098	0.003098
9	1/1/2015	Quarter1	1	3	1	0.75	28.08	681.0	6900	45	0.0	0	0	57.5	0.750428	0.000428
X = df[['team', 'targeted_productivity', 'day', 'department']]
y1 = df['productivity_difference']
y2 = df['actual_productivity']
# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = model.predict(X_test)
mse_linear_regressor = mean_squared_error(y_test, y_pred)
r2_linear_regressor = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse_linear_regressor}")
print(f"R-squared: {r2_linear_regressor}")
Mean Squared Error: 0.022790260180143244
R-squared: 0.03007778203078537
Decision tree regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
DecisionTreeRegressor()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = model.predict(X_test)
mse_decision_tree = mean_squared_error(y_test, y_pred)
r2_decision_tree = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse_decision_tree}")
print(f"R-squared: {r2_decision_tree}")
Mean Squared Error: 0.031002022325531748
R-squared: -0.3194035530015553
## Support Vector Regressor
model = SVR()
model.fit(X_train, y_train)
SVR()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = model.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred)
r2_svr = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse_svr}")
print(f"R-squared: {r2_svr}")
Mean Squared Error: 0.02226208849116378
R-squared: 0.05255604476206632
Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
RandomForestRegressor()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = model.predict(X_test)
mse_random_forest = mean_squared_error(y_test, y_pred)
r2_random_forest = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse_random_forest}")
print(f"R-squared: {r2_random_forest}")
Mean Squared Error: 0.02490196161465726
R-squared: -0.05979333496669215
print(f"Mean squared error of Decision Tree: {mse_decision_tree}")
print(f"Mean squared error of Linear Regressor: {mse_linear_regressor}")
print(f"Mean squared error of SVR: {mse_svr}")
print(f"Mean squared error of Random Forest: {mse_random_forest}")
Mean squared error of Decision Tree: 0.031002022325531748
Mean squared error of Linear Regressor: 0.022790260180143244
Mean squared error of SVR: 0.02226208849116378
Mean squared error of Random Forest: 0.02490196161465726
print(f"R^2 of Decision Tree: {r2_decision_tree}")
print(f"R^2 of Linear Regressor: {r2_linear_regressor}")
print(f"R^2 of SVR: {r2_svr}")
print(f"R^2 of Random Forest: {r2_random_forest}")
R^2 of Decision Tree: -0.3194035530015553
R^2 of Linear Regressor: 0.03007778203078537
R^2 of SVR: 0.05255604476206632
R^2 of Random Forest: -0.05979333496669215
Support Vector Regressor looks like the best method to use for this particular case
model = SVR()
model.fit(X_train, y_train)
SVR()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
X_train.head()
team	targeted_productivity	day	department
1189	8	0.70	5	1
575	1	0.75	0	0
76	10	0.75	0	0
731	4	0.70	3	0
138	12	0.80	3	1
df.head(10)
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity	productivity_difference
0	1/1/2015	Quarter1	1	3	8	0.80	26.16	1108.0	7080	98	0.0	0	0	59.0	0.940725	0.140725
1	1/1/2015	Quarter1	0	3	1	0.75	3.94	0.0	960	0	0.0	0	0	8.0	0.886500	0.136500
2	1/1/2015	Quarter1	1	3	11	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
3	1/1/2015	Quarter1	1	3	12	0.80	11.41	968.0	3660	50	0.0	0	0	30.5	0.800570	0.000570
4	1/1/2015	Quarter1	1	3	6	0.80	25.90	1170.0	1920	50	0.0	0	0	56.0	0.800382	0.000382
5	1/1/2015	Quarter1	1	3	7	0.80	25.90	984.0	6720	38	0.0	0	0	56.0	0.800125	0.000125
6	1/1/2015	Quarter1	0	3	2	0.75	3.94	0.0	960	0	0.0	0	0	8.0	0.755167	0.005167
7	1/1/2015	Quarter1	1	3	3	0.75	28.08	795.0	6900	45	0.0	0	0	57.5	0.753683	0.003683
8	1/1/2015	Quarter1	1	3	2	0.75	19.87	733.0	6000	34	0.0	0	0	55.0	0.753098	0.003098
9	1/1/2015	Quarter1	1	3	1	0.75	28.08	681.0	6900	45	0.0	0	0	57.5	0.750428	0.000428
df.tail(10)
date	quarter	department	day	team	targeted_productivity	smv	wip	over_time	incentive	idle_time	idle_men	no_of_style_change	no_of_workers	actual_productivity	productivity_difference
1187	3/11/2015	Quarter2	1	5	4	0.75	26.82	1054.0	7080	45	0.0	0	0	59.0	0.750051	0.000051
1188	3/11/2015	Quarter2	1	5	5	0.70	26.82	992.0	6960	30	0.0	0	1	58.0	0.700557	0.000557
1189	3/11/2015	Quarter2	1	5	8	0.70	30.48	914.0	6840	30	0.0	0	1	57.0	0.700505	0.000505
1190	3/11/2015	Quarter2	1	5	6	0.70	23.41	1128.0	4560	40	0.0	0	1	38.0	0.700246	0.000246
1191	3/11/2015	Quarter2	1	5	7	0.65	30.48	935.0	6840	26	0.0	0	1	57.0	0.650596	0.000596
1192	3/11/2015	Quarter2	0	5	10	0.75	2.90	0.0	960	0	0.0	0	0	8.0	0.628333	-0.121667
1193	3/11/2015	Quarter2	0	5	8	0.70	3.90	0.0	960	0	0.0	0	0	8.0	0.625625	-0.074375
1194	3/11/2015	Quarter2	0	5	7	0.65	3.90	0.0	960	0	0.0	0	0	8.0	0.625625	-0.024375
1195	3/11/2015	Quarter2	0	5	9	0.75	2.90	0.0	1800	0	0.0	0	0	15.0	0.505889	-0.244111
1196	3/11/2015	Quarter2	0	5	6	0.70	2.90	0.0	720	0	0.0	0	0	6.0	0.394722	-0.305278
# Collect user input (you can customize this part as needed)
user_input = []
user_input.append(float(input("Enter team: ")))
user_input.append(float(input("Enter targeted_productivity:  ")))
user_input.append(float(input("Enter day:  ")))
user_input.append(float(input("Enter department:  ")))

user_input = np.array(user_input).reshape(1, -1)

# Make predictions
prediction = model.predict(user_input)
print("Actual productivity = ", prediction + user_input[0, 1])
Actual productivity =  [85.05605975]
c:\Users\gayat\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but SVR was fitted with feature names
  warnings.warn(
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:\\Users\\gayat\\Downloads\\garments_worker_productivity (1).csv")

# Encode categorical columns
le = LabelEncoder()
df['department'] = le.fit_transform(df['department'])
df['day'] = le.fit_transform(df['day'])

# Prepare the data for daily predictions
X = df[['team', 'targeted_productivity', 'day', 'department']]
y = df['actual_productivity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model for daily predictions
best_model = RandomForestRegressor(random_state=42)  # Add a fixed random_state for reproducibility
best_model.fit(X_train, y_train)

# Prepare the data for monthly predictions
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
y_monthly = df.groupby('month')['actual_productivity'].mean()
X_monthly = df.groupby('month')[['team', 'targeted_productivity', 'day', 'department']].mean()

# Split the monthly data into training and testing sets for monthly predictions
X_train_monthly, X_test_monthly, y_train_monthly, y_test_monthly = train_test_split(X_monthly, y_monthly, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model for monthly predictions
best_model_monthly = RandomForestRegressor(random_state=42)  # Add a fixed random_state for reproducibility
best_model_monthly.fit(X_train_monthly, y_train_monthly)

# Collect user input for daily prediction
user_input_daily = []
user_input_daily.append(float(input("Enter team for daily prediction: ")))
user_input_daily.append(float(input("Enter targeted productivity for daily prediction: ")))
user_input_daily.append(float(input("Enter day for daily prediction: ")))
user_input_daily.append(float(input("Enter department for daily prediction: ")))

user_input_daily = np.array(user_input_daily).reshape(1, -1)

# Make prediction for daily productivity
daily_prediction = best_model.predict(user_input_daily)
print("Predicted daily productivity:", daily_prediction[0])

# Collect user input for monthly prediction
month_input = input("Enter month for monthly prediction (YYYY-MM): ")

# Validate month input
try:
    month_input_period = pd.Period(month_input)
    if month_input_period in X_monthly.index:
        monthly_input = X_monthly.loc[month_input_period].values.reshape(1, -1)
        monthly_input_df = pd.DataFrame(monthly_input, columns=X_monthly.columns)
        monthly_prediction = best_model_monthly.predict(monthly_input_df)
        print("Predicted monthly productivity:", monthly_prediction[0])  # No need to multiply by 100 if it's already percentage
    else:
        print("Error: The entered month is not available in the dataset.")
except Exception as e:
    print("Error: Invalid month format. Please enter in 'YYYY-MM' format.")
    print(e)

# Evaluate model performance on test sets
y_pred_monthly_test = best_model_monthly.predict(X_test_monthly)
mse_monthly_test = mean_squared_error(y_test_monthly, y_pred_monthly_test)
r2_monthly_test = r2_score(y_test_monthly, y_pred_monthly_test)
print(f"Mean Squared Error (Monthly - Test Set): {mse_monthly_test}")
print(f"R-squared (Monthly - Test Set): {r2_monthly_test}")
c:\Users\gayat\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
  warnings.warn(
Predicted daily productivity: 0.8233309901376671
Predicted monthly productivity: 0.7182998500212122
Mean Squared Error (Monthly - Test Set): 0.0013775396521346888
R-squared (Monthly - Test Set): nan
c:\Users\gayat\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\metrics\_regression.py:1187: UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
  warnings.warn(msg, UndefinedMetricWarning)
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("C:\\Users\\gayat\\Downloads\\garments_worker_productivity (1).csv")

# Encode categorical columns
le = LabelEncoder()
df['department'] = le.fit_transform(df['department'])
df['day'] = le.fit_transform(df['day'])

# Prepare the data for daily predictions
X = df[['team', 'targeted_productivity', 'day', 'department']]
y = df['actual_productivity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model for daily predictions
best_model = RandomForestRegressor(random_state=42)  # Add a fixed random_state for reproducibility
best_model.fit(X_train, y_train)

# Prepare the data for monthly predictions
df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
y_monthly = df.groupby('month')['actual_productivity'].mean()
X_monthly = df.groupby('month')[['team', 'targeted_productivity', 'day', 'department']].mean()

# Split the monthly data into training and testing sets for monthly predictions
X_train_monthly, X_test_monthly, y_train_monthly, y_test_monthly = train_test_split(X_monthly, y_monthly, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model for monthly predictions
best_model_monthly = RandomForestRegressor(random_state=42)  # Add a fixed random_state for reproducibility
best_model_monthly.fit(X_train_monthly, y_train_monthly)

# Function to predict daily productivity
def predict_daily(team, targeted_productivity, day, department):
    user_input_daily = np.array([team, targeted_productivity, day, department]).reshape(1, -1)
    daily_prediction = best_model.predict(user_input_daily)
    return daily_prediction[0]

# Function to predict monthly productivity
def predict_monthly(month):
    try:
        month_input_period = pd.Period(month)
        if month_input_period in X_monthly.index:
            monthly_input = X_monthly.loc[month_input_period].values.reshape(1, -1)
            monthly_input_df = pd.DataFrame(monthly_input, columns=X_monthly.columns)
            monthly_prediction = best_model_monthly.predict(monthly_input_df)
            return monthly_prediction[0]  # No need to multiply by 100 if it's already percentage
        else:
            return "Error: The entered month is not available in the dataset."
    except Exception as e:
        return f"Error: Invalid month format. Please enter in 'YYYY-MM' format. Error details: {e}"

# Create interfaces for daily and monthly predictions
daily_interface = gr.Interface(fn=predict_daily, inputs=["text", "text", "text", "text"], outputs="text", 
                               title="Daily Productivity Prediction",
                               description="Enter team, targeted productivity, day, and department for daily prediction.")

monthly_interface = gr.Interface(fn=predict_monthly, inputs="text", outputs="text", 
                                 title="Monthly Productivity Prediction",
                                 description="Enter month (YYYY-MM) for monthly prediction.")

# Launch interfaces
daily_interface.launch()
monthly_interface.launch()
c:\Users\gayat\AppData\Local\Programs\Python\Python311\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Running on local URL:  http://127.0.0.1:7861

To create a public link, set `share=True` in `launch()`.
IMPORTANT: You are using gradio version 3.44.4, however version 4.29.0 is available, please upgrade.
--------
IMPORTANT: You are using gradio version 3.44.4, however version 4.29.0 is available, please upgrade.
--------
c:\Users\gayat\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
  warnings.warn(
