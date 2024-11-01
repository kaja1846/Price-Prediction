import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Kaja\\price-prediction\\data\\rental_1000.csv")

# Separate features and target variable
X = df[['rooms', 'sqft']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# New data for prediction
new_data = pd.DataFrame({
    'rooms': [2, 4],
    'sqft': [1125, 1334]
})
# Predict the price using the trained model
predictions = model.predict(new_data)

# Output the predictions
print("Predicted prices:", predictions)