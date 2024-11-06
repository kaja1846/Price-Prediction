from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Flask app
app = Flask(__name__)

# Load the data
df = pd.read_csv("/app/data/rental_1000.csv")

# Separate features and target variable
X = df[['rooms', 'sqft']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional, useful for logs)
mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_data = pd.DataFrame(data)
    predictions = model.predict(new_data)
    return jsonify({'predictions': predictions.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
