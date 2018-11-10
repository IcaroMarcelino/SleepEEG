from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier as MLP
import csv
import random
import numpy as np
from input_output import*

files = ['data/data_75/wav75_ex1_.csv', 'data/data_75/wav75_ex2_.csv', 'data/data_75/wav75_ex3_.csv',
		'data/data_75/wav75_ex4_.csv', 'data/data_75/wav75_ex5_.csv', 'data/data_75/wav75_ex6_.csv',
		'data/data_75/wav75_ex7_.csv', 'data/data_75/wav75_ex8_.csv']

output = open('Analysis/MLP_13_3.csv', 'w') 
output.write("Balanced,#Neurons,Activattion,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc,AUC0,AUC1\n")

for act in ['relu', 'tanh']:
	for n_neurons in [15,30,45,60,75]:
		for balance in [1,0]:
			print(act, n_neurons, balance)
			for i in range(10):
				print(i)
				X_train1, y_train1, X_test1, y_test1, clss = import_all_data(files,1,.3,balance,not(balance))
				classifier = MLP(hidden_layer_sizes=(n_neurons, ), activation=act, solver='adam', learning_rate = 'constant', learning_rate_init = 0.09,momentum = 0.65,max_iter = 200)
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

				output.write(str(balance) + ',' + str(n_neurons) + ',' + act + ',' + str(i) + ',' + str(prf[0][0]) + ',' 
					+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
					+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
					+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
					+ str(acc) + ',' + str(AUC[0]) + ',' + str(AUC[1]) + '\n')
output.close()

