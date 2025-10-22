# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Sample data
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 35, 40],
    'Salary': [15000, 29000, 48000, 60000, 52000, 61000, 58000, 70000, 35000, 42000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 0, 1]  # 0 = No, 1 = Yes
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Split into features and target
X = df[['Age', 'Salary']]
y = df['Purchased']

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Optional: Predict for a new user
new_user = [[30, 40000]]  # Age = 30, Salary = 40000
prediction = model.predict(new_user)
print("Purchased" if prediction[0] == 1 else "Not Purchased")
