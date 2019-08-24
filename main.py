import pandas as pd
import preProcess as pp
import learningAlgorithm as algo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('Youtube-KatyPerry.csv')

df = df.drop('AUTHOR', axis=1)
df = df.drop('COMMENT_ID', axis=1)
df = df.drop('DATE', axis=1)

X = pp.word2vec(df)
Y = df['CLASS']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

model = algo.decesionTree(X_train, Y_train)
Y_pred = model.predict(X_test)

report = classification_report(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)

print("Classification Report\n", report)
print("Accuracy is ", round(accuracy*100, 2), "%")
