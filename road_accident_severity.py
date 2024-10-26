import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Install seaborn if not already installed
import subprocess
import sys

try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

# Function to load and use the saved model for predictions
def predict_severity(vehicles_involved, weather_conditions):
    model = joblib.load('accident_severity_model.pkl')
    example_data = np.array([[vehicles_involved, weather_conditions]])
    predicted_severity = model.predict(example_data)
    return predicted_severity

# Load the dataset
df = pd.read_csv('data1.csv')

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Check for NaN values and handle them
if df.isnull().values.any():
    print("Warning: The dataset contains NaN values. Please handle them before proceeding.")
    # You can drop NaN values or fill them as appropriate
    # df = df.dropna()  # Uncomment this line to drop NaN values
    # df = df.fillna(method='ffill')  # Uncomment this line to fill NaN values

# Encode categorical variables if necessary
df['weather_conditions'] = df['weather_conditions'].astype('category').cat.codes

# Define dependent and independent variables
X = df[['vehicles_involved', 'weather_conditions']]  # Adjust based on actual column names
y = df['severity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

# Example prediction
example_data = np.array([[3, 1]])  # Example: 3 vehicles, clear weather (encoded as 1)
predicted_severity = model.predict(example_data)
print('Predicted Accident Severity:', predicted_severity)

# Generate multiple predictions for visualization
vehicles_range = np.arange(1, 10)  # Example range for number of vehicles
weather_conditions_range = np.arange(0, 3)  # Example range for weather conditions (encoded)

predictions = []
for vehicles in vehicles_range:
    for weather in weather_conditions_range:
        predictions.append((vehicles, weather, predict_severity(vehicles, weather)[0]))

predictions_df = pd.DataFrame(predictions, columns=['vehicles_involved', 'weather_conditions', 'predicted_severity'])

# Create a single figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot Actual vs Predicted
axs[0, 0].scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
axs[0, 0].set_xlabel('Actual Accident Severity')
axs[0, 0].set_ylabel('Predicted Accident Severity')
axs[0, 0].set_title('Actual vs Predicted Accident Severity')

# Calculate the line of best fit
z = np.polyfit(y_test, y_pred, 1)  # 1 for linear fit
p = np.poly1d(z)
axs[0, 0].plot(y_test, p(y_test), color='red', label='Line of Best Fit')

axs[0, 0].legend()

# Plot Residuals Distribution
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, ax=axs[0, 1])
axs[0, 1].set_xlabel('Residuals')
axs[0, 1].set_title('Distribution of Residuals')

# Plot Residuals vs Predicted
axs[1, 0].scatter(y_pred, residuals)
axs[1, 0].axhline(y=0, color='r', linestyle='--')
axs[1, 0].set_xlabel('Predicted Accident Severity')
axs[1, 0].set_ylabel('Residuals')
axs[1, 0].set_title('Residuals vs Predicted Accident Severity')

# Plot Multiple Predictions
sns.scatterplot(data=predictions_df, x='vehicles_involved', y='predicted_severity', hue='weather_conditions', palette='viridis', ax=axs[1, 1])
axs[1, 1].set_xlabel('Number of Vehicles Involved')
axs[1, 1].set_ylabel('Predicted Accident Severity')
axs[1, 1].set_title('Predicted Accident Severity for Various Conditions')
axs[1, 1].legend(title='Weather Conditions')

plt.tight_layout()
plt.show()

# Display model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Example usage of the function
print('Predicted Accident Severity for 3 vehicles and clear weather:', predict_severity(3, 1))

# Create a pie chart based on the metrics
fig, ax = plt.subplots(figsize=(8, 8))
metrics = ['Mean Squared Error', 'R^2 Score', 'Average Predicted Severity']
values = [mse, r2, np.mean(predictions_df['predicted_severity'])]

# Normalize values for pie chart
total = sum(values)
if total > 0:  # Ensure we don't divide by zero
    values = [v / total for v in values]

ax.pie(values, labels=metrics, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Performance Metrics Distribution')
plt.show()