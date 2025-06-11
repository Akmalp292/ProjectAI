import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import csv

# Load the dataset
with open("./ASL.pickle", "rb") as f:
    dataset = pickle.load(f)

# Convert dataset and labels to numpy arrays
data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

# Load the trained model
with open("./ASL_model.p", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

# Evaluate the model on the test data
y_pred = model.predict(X_test)

# Calculate and print the accuracy score
score = accuracy_score(y_pred, y_test)
print(f"Accuracy Score: {score}")

# Display the classification report for detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the results to a CSV file
csv_file_path = "predictions.csv"

# Write the results to CSV
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Actual Label", "Predicted Label"])

    # Write the actual and predicted labels for each test sample
    for actual, predicted in zip(y_test, y_pred):
        writer.writerow([actual, predicted])

print(f"\nPredictions saved to {csv_file_path}")

# Extract precision, recall, and f1-score from classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Display the precision, recall, and f1-score for each class
print("\nClass-wise Precision, Recall, and F1-Score:")
print(f"{'Class':<20}{'Precision':<12}{'Recall':<12}{'F1-Score'}")
print("-" * 50)

# Iterate through each class and print the precision, recall, and f1-score
for class_label in report.keys():
    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip non-class metrics
        precision = report[class_label]['precision']
        recall = report[class_label]['recall']
        f1_score = report[class_label]['f1-score']
        print(f"{class_label:<20}{precision:<12.4f}{recall:<12.4f}{f1_score:<12.4f}")
