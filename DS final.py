# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
data = pd.read_csv(r"C:\Users\Tejas Holkar\Desktop\boxoffice.csv")

# 1. Data Preprocessing: Dropping columns we won't use for prediction
df = data[['domestic_revenue', 'world_revenue', 'opening_revenue', 'budget', 'opening_theaters', 'release_days']]

# 2. Visualizing Data
# Box Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    plt.boxplot(df[col], patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Histograms
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    plt.hist(df[col], bins=20, color="skyblue", edgecolor="black")
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(['budget', 'domestic_revenue', 'opening_revenue'], 1):
    plt.subplot(2, 2, i)
    plt.scatter(df[col], df['world_revenue'], color="blue", alpha=0.6)
    plt.title(f'Scatter Plot of {col} vs World Revenue')
    plt.xlabel(col)
    plt.ylabel('World Revenue')
plt.tight_layout()
plt.show()

# Correlation Matrix (Using Pandas)
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize Correlation Matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()

# 4. Identifying Dependent and Independent Features
# The dependent variable (Y) is 'world_revenue'
X = df[['domestic_revenue', 'opening_revenue', 'budget', 'opening_theaters', 'release_days']].values
y = df['world_revenue'].values

# 5. Splitting the data into training and testing sets manually (80% train, 20% test)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 6. Implementing Linear Regression Manually
# Add a column of ones for the bias term (intercept)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Calculate coefficients using the Normal Equation: theta = (X'X)^-1 X'y
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# 7. Predict the world_revenue using the test data
y_pred = X_test @ theta

# 8. Model Evaluation: Calculate Mean Squared Error
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

# 9. Display the coefficients of the model
coefficients = pd.DataFrame(theta, index=['Intercept'] + list(df.columns[:-1]), columns=['Coefficient'])
print("Coefficients of the model:")
print(coefficients)

# Calculate Pearson Correlation between domestic_revenue and budget
list1 = data["domestic_revenue"]
list2 = data["budget"]
corr, _ = pearsonr(list1, list2)

print(f"Pearson Correlation between Domestic Revenue and Budget: {corr}")
