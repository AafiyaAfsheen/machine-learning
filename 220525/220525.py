import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Loading data
df = pd.read_csv('220525/Titanic-Dataset.csv')
print("First 10 rows:", df.head(10))
print("Shape of dataset:", df.shape)
print("Info:", df.info())

print("Summary statistics:v", df.describe())

# 2. missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)

# 3. Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 4. Age 
df['AgeGroup'] = pd.cut(df['Age'], bins=5, labels=False)

# 5. One-hot encode
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 6. Dropping irrelevant columns
df.drop(['PassengerId', 'Ticket', 'Name'], axis=1, inplace=True)

# 7. Visualizations
plt.figure(figsize=(12, 4))

# Histogram 
plt.subplot(1, 3, 1)
plt.hist(df['Age'], bins=20, color='skyblue')
plt.title('Histogram of Age')

# Bar chart
plt.subplot(1, 3, 2)
df.groupby('Sex_male')['Survived'].sum().plot(kind='bar', color='salmon')
plt.title('Survival Count by Gender')
plt.xticks([0, 1], ['Female', 'Male'])

# Box plot
plt.subplot(1, 3, 3)
df.boxplot(column='Fare', by='Pclass')
plt.title('Fare by Pclass')
plt.suptitle('')

plt.tight_layout()
plt.show()

# 8.analysis
fare_array = df['Fare'].values
age_array = df['Age'].values

print("\nFare - Mean:", np.mean(fare_array), "Median:", np.median(fare_array), "Std:", np.std(fare_array))
print("Age - Mean:", np.mean(age_array), "Median:", np.median(age_array), "Std:", np.std(age_array))
df['Fare_norm'] = (fare_array - np.min(fare_array)) / (np.max(fare_array) - np.min(fare_array))
df['Age_norm'] = (age_array - np.min(age_array)) / (np.max(age_array) - np.min(age_array))

# 9. Correlation matrix and heatmap
corr_matrix = df.select_dtypes(include=np.number).corr()
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Matrix Heatmap')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout()
plt.show()

# 10. Split into X and y
X = df.drop('Survived', axis=1)
y = df['Survived']
processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv('220525/titanic_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")


