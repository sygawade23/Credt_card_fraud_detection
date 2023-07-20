import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the credit card dataset into a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')

# Displaying the distribution of legitimate transactions (Class=0) and fraudulent transactions (Class=1)
credit_card_data['Class'].value_counts()

# Separating the data into two DataFrames: one for legitimate transactions (legit) and one for fraudulent transactions (fraud)
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print(legit.shape)  # Output the shape of the legit DataFrame (number of rows and columns)
print(fraud.shape)  # Output the shape of the fraud DataFrame (number of rows and columns)

# Displaying statistical measures of the 'Amount' column for both legit and fraud transactions
legit.Amount.describe()
fraud.Amount.describe()

# Comparing the mean values for each feature between legit and fraud transactions
credit_card_data.groupby('Class').mean()

# Sampling a subset of legit transactions (492 samples) to balance the dataset and creating a new dataset
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Displaying the distribution of Class column in the new dataset to verify the balance
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

# Separating the Class column for prediction and evaluation
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Splitting the data into training and testing sets with 20% as the test size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Creating a Logistic Regression model
model = LogisticRegression()

# Training the Logistic Regression model with the training data
model.fit(X_train, Y_train)

# Calculating accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# Calculating accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
