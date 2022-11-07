# FoML Assign 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Enter You Name Here
myname = "Tushar_K_Raysad" # or "Amar-Akbar-Antony"

# Implement your decision tree below
class DecisionTree():
    tree = {}
















    def learn(self, training_set):
        # implement this function
        self.tree = {} 
        df = pd.DataFrame(training_set)
        X_train = df.drop(11, axis=1)
        y_train = df[[11]]
       
        clf_model = DecisionTreeClassifier()
        clf_model.fit(X_train, y_train)
        # print(clf_model)
        self.tree["model"]=clf_model
        # print(tree)
        

    # implement this function
    def classify(self, test_instance):
        nparr = np.asarray(test_instance).reshape(1, -1)
        
        result = self.tree.get("model").predict(nparr)
        return result

def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print ("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    test_set = [x for i, x in enumerate(data) if i % K == 9]
    
    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn( training_set )
    
    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print( "accuracy: %.4f" % accuracy)       
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "a")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
