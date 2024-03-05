import pandas as pd
import numpy as np

# Define the number of records to generate
num_records = 1000

# Generate random values for the features
exercise = np.random.randint(low=1, high=5, size=num_records)
sleep = np.random.randint(low=4, high=10, size=num_records)
diet = np.random.randint(low=1, high=5, size=num_records)
social_interactions = np.random.randint(low=1, high=5, size=num_records)
leisure_activities = np.random.randint(low=1, high=5, size=num_records)

# Generate random values for the target variable
stress_levels = np.random.choice(['low', 'moderate', 'high'], size=num_records)

# Combine the features and target variable into a single DataFrame
data = pd.DataFrame({'Exercise': exercise,
                     'Sleep': sleep,
                     'Diet': diet,
                     'Social Interactions': social_interactions,
                     'Leisure Activities': leisure_activities,
                     'Stress Level': stress_levels})

# Write the DataFrame to a CSV file
data.to_csv('synthetic_data.csv', index=False)
