import pandas as pd
import numpy as np
import math
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

class NB:
    def fit(self, X, y):
        s, f = X.shape
        self.cla = np.unique(y)
        c = len(self.cla)
        self.prior = np.zeros(c, dtype=np.float64)
        self.mean_val = np.zeros((c, f), dtype=np.float64)
        self.varr = np.zeros((c, f), dtype=np.float64)
        for i, j in enumerate(self.cla):
            alterX = X[y == j]
            self.prior[i] = alterX.shape[0] / float(s)
            self.varr[i, :] = alterX.var(axis=0)
            self.mean_val[i, :] = alterX.mean(axis=0)

    def predict(self, X):
        y_pred = [self.pred(x) for x in X]
        return y_pred

    def pred(self, x):
        post = []
        for i, j in enumerate(self.cla):
            posterior = np.sum(np.log(self.pad(i, x)))
            prior = np.log(self.prior[i])
            posterior = prior + posterior
        post.append(posterior)
        return self.cla[np.argmax(post)]

    def pad(self, i, x):
        var = self.varr[i]
        mean = self.mean_val[i]
        nu = np.exp(-((x - mean) ** 2) / (2 * var))
        den = np.sqrt(2 * np.pi * var)
        return nu / den


bcd = pd.read_csv('./dataset/breast-cancer-wisconsin.data', sep=',', header=None)
bcd.columns = ['Sample code', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
'Normal Nucleoli', 'Mitoses','Class']
bcd = bcd.drop(['Sample code'],axis=1)
bcd = bcd.replace('?',np.NaN)
bcd_ = bcd['Bare Nuclei']
bcd_ = bcd_.fillna(bcd_.median())
bcd_ = bcd.dropna()
x = bcd_.iloc[:,:-1].values.astype(int)
y = bcd_.iloc[:,-1].values.astype(int)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def variance(x):
    n = len(x)
    mean = sum(x) / n
    deviations = [(x - mean) ** 2 for x in x]
    variance = sum(deviations) / n
    return variance


def standard_Deviation(x):
    var = variance(x)
    print(var)
    std_dev = math.sqrt(var)
    return std_dev


cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
scores=[]

for i, j in cv.split(x,y):
    X_train, X_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
    nb = NB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    score=accuracy(y_test, predictions)
    scores.append(score)
print('Accuracy: %.2f%%' % (sum(scores)/float(len(scores))))
print('Standard Deviation: % s' % (standard_Deviation(scores)))

cd = pd.read_csv('./dataset/car.data', sep=',', header=0)
cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
cd.columns = cols
X = cd.drop(['class'], axis=1)
y = cd['class']
cd = cd.dropna()
cd.reset_index(drop=True, inplace=True)

encoder_X = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
x= encoder_X.fit_transform(X)
encoder_Y = ce.OrdinalEncoder()
y = np.ravel(encoder_Y.fit_transform(y))

x = pd.DataFrame(x)
y = pd.DataFrame(y)
y.columns=['class']
x = x.values.astype(int)
y = y.iloc[:,0].values.astype(int)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
scores=[]

for i, j in cv.split(x,y):
    X_train, X_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
    nb = NB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    score=accuracy(y_test, predictions)
    scores.append(score)
print('Accuracy: %.2f%%' % (sum(scores)/float(len(scores))))
print('Standard Deviation: % s' % (standard_Deviation(scores)))


md = pd.read_csv("./dataset/mushroom.data", sep=',', header=None)
md.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
'stalk-surface-below-ring', 'stalk-color-above-ring',
'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
'ring-type', 'spore-print-color', 'population', 'habitat']
col=['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
'stalk-surface-below-ring', 'stalk-color-above-ring',
'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
'ring-type', 'spore-print-color', 'population', 'habitat']
labelencoder=LabelEncoder()
for column in md.columns:
    md[column] = labelencoder.fit_transform(md[column])
x = md.iloc[:,1:].values.astype(int)
y= md.iloc[:,0].values.astype(int)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=3)
scores=[]

for i, j in cv.split(x,y):
    X_train, X_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
    nb = NB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    score=accuracy(y_test, predictions)
    scores.append(score)
print('Accuracy: %.2f%%' % (sum(scores)/float(len(scores))))
print('Standard Deviation: % s' % (standard_Deviation(scores)))

ed = pd.read_csv("./dataset/ecoli.data",header=None,sep="\s+")
col_names = ["squence_name","mcg","gvh","lip","chg","aac","alm1","alm2","site"]
ed.columns = col_names
ed.loc[:,ed.dtypes == "object"].columns.tolist()

def cleaning_object(ed,cols_to_drop,class_col):
    ed = ed.drop(cols_to_drop,axis=1)
    uni_class = ed[class_col].unique().tolist()
    for class_label in uni_class:
        num_rows = sum(ed[class_col] == class_label)
        if num_rows < 10:
            class_todrop = ed[ed[class_col] == class_label].index
        ed.drop(class_todrop,inplace = True)
    return ed

ced = cleaning_object(ed,["squence_name",'lip','chg'],"site")
ced["site"].value_counts()
encoder_Y = ce.OrdinalEncoder()
y=np.ravel(encoder_Y.fit_transform(ced["site"]))
ced=ced.drop('site',axis=1,)
X = ced.values.astype(float)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=123)
scores=[]

for i, j in cv.split(x,y):
    X_train, X_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
    nb = NB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    score=accuracy(y_test, predictions)
    scores.append(score)
print('Accuracy: %.2f%%' % (sum(scores)/float(len(scores))))
print('Standard Deviation: % s' % (standard_Deviation(scores)))

ld = pd.read_csv("./dataset/letter-recognition.data",header=None)
col_names = ["letter","xbox","ybox","width","height","onpix","xbar","ybar","x2bar","y2bar","xybar","x2ybar","xy2bar","xedge","xedgey","yedge","yedgex"]
ld.columns = col_names
X = ld.iloc[:, 1:]
y = ld['letter'].tolist()
x = X.values.astype(float)
encoder_Y = ce.OrdinalEncoder()
y=np.ravel(encoder_Y.fit_transform(y))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
scores=[]

for i, j in cv.split(x,y):
    X_train, X_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
    nb = NB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    score=accuracy(y_test, predictions)
    scores.append(score)
print('Accuracy: %.2f%%' % (sum(scores)/float(len(scores))))
print('Standard Deviation: % s' % (standard_Deviation(scores)))