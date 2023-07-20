# Credt_card_fraud_detection
This code performs the following steps:

Loads the credit card dataset into a Pandas DataFrame and examines the distribution of legitimate and fraudulent transactions.
Separates the data into two DataFrames: one for legitimate transactions and one for fraudulent transactions.
Compares the statistical measures and mean values of features between legitimate and fraudulent transactions to observe any differences.
Creates a balanced dataset by randomly sampling a subset of legitimate transactions to match the number of fraudulent transactions.
Splits the balanced dataset into training and testing sets (80% for training and 20% for testing).
Trains a Logistic Regression model on the training data.
Evaluates the model's accuracy on both the training and test data.
It's important to note that the logistic regression model is not the only possible model for fraud detection, and other more advanced techniques like anomaly detection or ensemble methods could also be used. Additionally, evaluating the model's performance with accuracy might not be sufficient, and other metrics such as precision, recall, or F1-score should also be considered, especially in imbalanced datasets like this one.
