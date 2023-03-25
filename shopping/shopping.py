import csv
import sys
import pandas as pd
import numpy as numpy

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    #Getting the data
    data = pd.read_csv("shopping.csv", header=0)

    #Dictionary storing months with 0-index values
    d = {"Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5, "Jul":6, "Aug":7, "Sep":8, "Oct":9, "Nov":10, "Dec":11}

    #Changing months to int values
    data.Month = data.Month.map(d)

    #Changing visitor values to 1 for returning visitor and 0 for non-returning visitor
    data.VisitorType = data.VisitorType.map(lambda x : 1 if x=="Returning_Visitor" else 0)

    #Changing boolean values in Weekend to 1 or 0
    data.Weekend = data.Weekend.map(lambda x : 1 if x==True else 0)

    #Changing boolean values in Revenue to 1 or 0
    data.Revenue = data.Revenue.map(lambda x : 1 if x==True else 0)

    #Definning int and float columns for check-up of dtype
    ints = ["Administrative", "Informational", "ProductRelated", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    floats = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]

    #Checking for the type of int columns and convert them into non-int type
    for value in ints:
        if data[value].dtype != "int64":
            data = data.astype({value: "int64"})
        else:
            continue

    #Checking for the type of int columns and convert them into non-int type
    for value in floats:
        if data[value].dtype != "float64":
            data = data.astype({value: "float64"})
        else:
            continue

    #Creating a list of lists from dataframe on evidence-values
    evidence = data.iloc[:,:-1].values.tolist()
    #Creating a list of lists from last columns(series) on label-values
    labels = data.iloc[:,-1].values.tolist()

    #Checking if evidence and labels have the same length
    if len(evidence) != len(labels):
        print("Error! Evidence and label lists do not have the same length, Check code!")
    else:
        print(f"There are {len(evidence)} entries in this dataset.\n")

    #Returning the evidence and labels
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #Creating a model with the K-Nearest Neighbors
    model = KNeighborsClassifier(n_neighbors=1)

    #Training the model
    model.fit(evidence, labels)

    #Returning the model
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    #Getting the number of actual positives and negatives from label
    positives = labels.count(1)
    negatives = labels.count(0)

    #Initiate sensitivity and specitivity variables
    sens = 0
    spec = 0

    #Iterate over actual labels and predicted labels at the same time
    for label, pred in zip(labels, predictions):
        if label == 1:
            #If prediction correct, increase sensitivity counter by 1
            if label == pred:
                sens += 1
        else:
            if label == pred:
                #If prediction correct, increase specitivity counter by 1
                spec += 1

    #Calculate portion of correct positive and negatives
    sensitivity = sens / positives
    specificity = spec / negatives

    #Returning values
    return(sensitivity, specificity)

if __name__ == "__main__":
    main()