import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate Synthetic Credit Data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'income': np.random.normal(55000, 15000, n),
    'loan_amount': np.random.normal(15000, 5000, n),
    'credit_score': np.random.randint(300, 850, n),
    'years_at_job': np.random.randint(0, 20, n)
})

# 2. Define 'Default' logic (The Target)
# High loan relative to income + low credit score = High risk of default (1)
data['default'] = ((data['loan_amount'] / data['income'] > 0.38) | (data['credit_score'] < 400)).astype(int)

# 3. Check the results
print("--- Dataset Head ---")
print(data.head())
print("\n--- Default Distribution (Percentage) ---")
print(data['default'].value_counts(normalize=True) * 100)

# 4. Visualize the relationship between Credit Score and Default
plt.figure(figsize=(10, 6))
sns.boxplot(x='default', y='credit_score', data=data, palette='viridis')
plt.title('Credit Score Distribution: Default (1) vs. Non-Default (0)')
plt.show()

# 5. Visualize Income vs. Loan Amount colored by Default
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='loan_amount', hue='default', data=data, alpha=0.7)
plt.title('Income vs. Loan Amount (Colored by Default Status)')
plt.axline((0, 0), slope=0.38, color='red', linestyle='--', label='Risk Threshold')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Prepare data for modeling
# X = Features (Income, Loan, Credit Score, etc.), y = Target (Default)
X = data.drop(['default'], axis=1)
y = data['default']

# 2. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make Predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Prepare data (X = Features, y = Target)
# We drop 'default' from X because that's what we want to predict
X = data.drop('default', axis=1)
y = data['default']

# 2. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Make Predictions
y_pred = clf.predict(X_test)

# 5. Evaluate the results
print("\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# 6. Feature Importance - What matters most?
importances = clf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance ---")
print(feature_importance_df)

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title('Which Factors Predict Default the Best?')
plt.show()

# Save the Importance plot
plt.savefig('feature_importance.png')

# Save the Scatter plot (Income vs Loan)
plt.savefig('risk_threshold_plot.png')