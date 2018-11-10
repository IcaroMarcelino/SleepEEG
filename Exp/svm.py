from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
import csv
import random
import numpy as np
from input_output import*

files = ['data/data_75/wav75_ex1_.csv', 'data/data_75/wav75_ex2_.csv', 'data/data_75/wav75_ex3_.csv',
		'data/data_75/wav75_ex4_.csv', 'data/data_75/wav75_ex5_.csv', 'data/data_75/wav75_ex6_.csv',
		'data/data_75/wav75_ex7_.csv', 'data/data_75/wav75_ex8_.csv']

output = open('Analysis2/SVM.csv', 'w') 
output.write("Balanced,Kernel,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc,AUC0,AUC1\n")

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
	for balance in [1,0]:
		print(kernel, balance)
		for i in range(100):
			print(i)
			X_train1, y_train1, X_test1, y_test1, clss = import_all_data(files,1,.3,balance,not(balance))
			classifier = SVC(kernel = kernel)
			y_train1 = np.array([j[1] for j in y_train1])
			y_test1 = np.array([j[1] for j in y_test1])
			
			classifier.fit(X_train1, y_train1)
			pred = classifier.predict(X_test1)
			acc = accuracy_score(y_test1, pred)
			prf = precision_recall_fscore_support(y_test1, pred)
			cfm = confusion_matrix(y_test1, pred).ravel()

			fpr = dict()
			tpr = dict()
			AUC = dict()
			pred = np.array([[int(not(j)),j] for j in pred])
			y_test1 = np.array([[int(not(j)),j] for j in y_test1])
			for c in range(2):
			    fpr[c], tpr[c], _ = roc_curve(y_test1[:, c], pred[:, c])
			    AUC[c] = auc(fpr[c], tpr[c])

			output.write(str(balance) + ',' + kernel +',' + str(i) + ',' + str(prf[0][0]) + ',' 
				+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
				+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
				+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
				+ str(acc) + ',' + str(AUC[0]) + ',' + str(AUC[1]) + '\n')
output.close()

