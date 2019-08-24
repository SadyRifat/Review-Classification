from sklearn.tree import DecisionTreeClassifier

def decesionTree(X_train, Y_train):
    dtree = DecisionTreeClassifier()
    return dtree.fit(X_train, Y_train)
