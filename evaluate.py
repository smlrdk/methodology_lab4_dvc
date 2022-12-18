import pandas as pd

from sklearn.metrics import accuracy_score

y_test = pd.read_csv('data/test/iris_test.csv')['class'].to_numpy().tolist()

kmeans_pred = list()
with open('k-means/predict.txt', 'r') as file:
    for line in file:
        kmeans_pred.append(int(line))

# dtree_pred = list()
# with open('dtree/predict.txt', 'r') as file:
#     for line in file:
#         dtree_pred.append(int(line))

# dtree_report = accuracy_score(y_test, dtree_pred)
kmeans_report = accuracy_score(y_test, kmeans_pred)

with open('metrics.txt', 'w') as file:
    file.write(f"k-means acc: {kmeans_report}")
    # file.write("\n")
    # file.write(f"dtree acc: {dtree_report}")