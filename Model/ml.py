import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('DataSet/synthetic_data.csv')

# Preprocess the data
data.dropna(inplace=True)
data = pd.get_dummies(data, columns=['Exercise', 'Sleep', 'Diet', 'Social Interactions', 'Leisure Activities'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('Stress Level', axis=1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['Stress Level'], test_size=0.2, random_state=42)

# Train a random forest classifier model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))

# Make personalized recommendations
user_data = pd.read_csv('synthetic_data.csv')
scaled_user_data = scaler.transform(pd.get_dummies(user_data, columns=['Exercise', 'Sleep', 'Diet', 'Social Interactions', 'Leisure Activities']).drop('Stress Level', axis=1))
user_stress_level = rf_model.predict(scaled_user_data)[0]
if user_stress_level == 'high':
    print('You should consider practicing stress-reducing techniques such as meditation or yoga.')
elif user_stress_level == 'moderate':
    print('You may benefit from incorporating more physical activity into your daily routine.')
else:
    print('You are doing a good job of managing your stress levels. Keep it up!')
