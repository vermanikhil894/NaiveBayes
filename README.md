# NaiveBayes
Classifier: A classifier is a machine learning model that is used to discriminate different objects based on certain features.

A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The Bayes theorem serves as the foundation of the classifier.

Bayes Theorem:
P(A|B) = (P(B|A)P(A))/P(B)

Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.

We have used the following datasets. The description of datasets can be found at:
1. Car.data -> https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
2. ecoli.data -> https://archive.ics.uci.edu/ml/datasets/Ecoli
3. mushroom.data -> https://archive.ics.uci.edu/ml/datasets/Mushroom
4. letter-recognition.data -> https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
5. breast-cancer-wisconsin.data -> https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)


Run below command to perform classification on the data:
python3 NaiveBayes.py
