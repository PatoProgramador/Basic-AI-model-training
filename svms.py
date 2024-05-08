from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

datos = datasets.load_breast_cancer(as_frame=True)

# print(datos.frame.isna().sum())

# print(datos.frame.describe())

# datos.data.shape

X = datos.data
Y = datos.target

X_ent,X_pru, Y_ent, Y_pru = train_test_split(X,Y, test_size=.5)

kernels = ["linear", "rbf", "sigmoid"]
gammas = [1, .01, .001, .0001, .00001]

better_accuracy_and_precision = {
    "accuracy": 0,
    "precision": 0,
    "kernel": "",
    "gamma": ""
}

for kernel in kernels:
    for gamma in gammas:
        model = svm.SVC(kernel=kernel, gamma=gamma)
        model.fit(X_ent, Y_ent)

        predi = model.predict(X_pru)
        
        accurracy = metrics.accuracy_score(Y_pru, predi)
        precision = metrics.precision_score(Y_pru, predi)
        
        if accurracy > better_accuracy_and_precision["accuracy"] and precision > better_accuracy_and_precision["precision"]:
            better_accuracy_and_precision["accuracy"] = accurracy
            better_accuracy_and_precision["precision"] = precision
            better_accuracy_and_precision["kernel"] = kernel
            better_accuracy_and_precision["gamma"] = gamma

        # print(f"{kernel}: Gamma: {gamma} Exactitud: ", accurracy)
        # print(f"{kernel}: Gamma: {gamma} Precision: ", precision)
        # print("*****************************************************")
        
print("Mejor perfomance")
print(better_accuracy_and_precision)
print("****************************************************************************")

b_kernel = better_accuracy_and_precision["kernel"]
b_gamma = better_accuracy_and_precision["gamma"]

model = svm.SVC(kernel= b_kernel, gamma=b_gamma)
model.fit(X_ent, Y_ent)

predi = model.predict(X_pru)
        
accurracy = metrics.accuracy_score(Y_pru, predi)
precision = metrics.precision_score(Y_pru, predi)

print(f"{b_kernel}: Gamma: {b_gamma} Exactitud: ", accurracy)
print(f"{b_kernel}: Gamma: {b_gamma} Precision: ", precision)
print("*********************** Classification report ******************************")
print(classification_report(Y_pru, predi))
print("*********************** Confussion matrix ******************************")
print(pd.DataFrame(confusion_matrix(Y_pru, predi)))