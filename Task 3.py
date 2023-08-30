import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

# Load the dataset
data = pd.read_csv('C:\\Users\\Master\\Desktop\\Datasets\\stocks.csv')
print(data.head())
print(data.dtypes)
print(data.isnull().sum())
data.fillna(data.median(numeric_only=True).round(1), inplace=True)

# Defining X and y
X = data[['open', 'high', 'low', 'volume']]
y = data['close']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

#Evaluation metrics
print("Linear Regression")
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error =",mse)
mae = mean_absolute_error(y_test,y_pred)
print("Mean absolute error =",mae)
r2 = r2_score(y_test, y_pred)
print("R squared =",r2)

#Random Forest
rf = RandomForestRegressor(n_estimators=50, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#Evaluation metrics
print("Random Forest")
mse2 = mean_squared_error(y_test, y_pred)
print("Mean Squared Error =",mse2)
mae2 = mean_absolute_error(y_test,y_pred)
print("Mean absolute error =",mae2)