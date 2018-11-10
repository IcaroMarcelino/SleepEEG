from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import csv
import random
import numpy as np
from input_output import*

files = [	'data/data_75/wav75_ex1_.csv', 'data/data_75/wav75_ex2_.csv', 'data/data_75/wav75_ex3_.csv',
				'data/data_75/wav75_ex4_.csv', 'data/data_75/wav75_ex5_.csv', 'data/data_75/wav75_ex6_.csv',
				'data/data_75/wav75_ex7_.csv', 'data/data_75/wav75_ex8_.csv']


output = open('Analysis2/KNN_1.csv', 'w') 
output.write("K,Balanced,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc,AUC0,AUC1\n")

for balance in [1,0]:
	for k in [3,5,7,9,11,13,15,17,19]:
		print(k)
		for i in range(10):
			print(i)
			X_train1, y_train1, X_test1, y_test1, clss = import_all_data(files,1,.3,balance,not(balance))
			knn = KNeighborsClassifier(n_neighbors=k)
			knn.fit(X_train1, y_train1)
			pred = knn.predict(X_test1)
			acc = accuracy_score(y_test1, pred)
			prf = precision_recall_fscore_support(y_test1, pred)
			cfm = confusion_matrix([j[0] for j in y_test1], [j2[0] for j2 in pred]).ravel()

			fpr = dict()
			tpr = dict()
			AUC = dict()
			for c in range(2):
			    fpr[c], tpr[c], _ = roc_curve(y_test1[:, c], pred[:, c])
			    AUC[c] = auc(fpr[c], tpr[c])

			output.write(str(k) + ',' + str(balance) + ',' + str(i) + ',' + str(prf[0][0]) + ',' 
				+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
				+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
				+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
				+ str(acc) + ',' + str(AUC[0]) + ',' + str(AUC[1]) + '\n')
output.close()

