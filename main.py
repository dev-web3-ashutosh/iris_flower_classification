# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=read_csv(url, names=names)

# check if dataset is loaded properly

# gives info of all attributes in details
# dataset.info()

# head; is different from .head
# print(dataset.head())
# print(dataset.head(20))

# rows x columns
# print(dataset.shape)

# statistical analysis
# print(dataset.describe())

# class distribution
# print(dataset.groupby('class').size())

# visualizations

# box and whisker (univariate)
'''
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
'''

# histogram (univariate)
'''
dataset.hist()
plt.show()
'''

# scatterplot; interactions between attributes (multivariate)
'''
scatter_matrix(dataset)
plt.show()
'''

# define input and output variables x and y
array=array(dataset)
x=array[:,:-1]
y=array[:,-1]

# split the data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=1)

# create models
models=[]

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate models
results=[]
names=[]

for name, model in models:
    kfold=StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results=cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

