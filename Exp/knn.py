from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import csv
import random
import numpy as np


files = ['data/wav_all_seg_ex1.csv', 	'data/wav_all_seg_ex2.csv', 'data/wav_all_seg_ex3.csv',
		'data/wav_all_seg_ex4.csv', 'data/wav_all_seg_ex5.csv', 'data/wav_all_seg_ex6.csv',
		'data/wav_all_seg_ex7.csv', 'data/wav_all_seg_ex8.csv']

def import_all_data(files_paths, rand, test_percent):
	total_x = []
	total_y = []
	class_ind = 0
	for file_path in files_paths:
		csvfile = open(file_path,'r')
		data = csv.reader(csvfile)
		X_S = []
		y_S = []
		X_NS = []
		y_NS = []
		for row in data:
			class_ind = len(row)-1
			if int(row[class_ind]):
				X_S.append([float(x) for x in row[0:class_ind]])
				y_S.append([int(row[class_ind]), int(not(int(row[class_ind])))])
			else:
				X_NS.append([float(x) for x in row[0:class_ind]])
				y_NS.append([int(row[class_ind]), int(not(int(row[class_ind])))])
		csvfile.close()
		temp1 = []
		temp2 = []
		index_shuf = list(range(len(X_NS)))
		random.shuffle(index_shuf)
		for i in index_shuf:
			temp1.append(X_NS[i])
			temp2.append(y_NS[i])
		X_train = temp1[0:len(X_S)] + X_S
		y_train = temp2[0:len(X_S)] + y_S
		if rand:
			temp1 = []
			temp2 = []
			index_shuf = list(range(len(X_train)))
			random.shuffle(index_shuf)
			for i in index_shuf:
				temp1.append(X_train[i])
				temp2.append(y_train[i])
			X_train = temp1
			y_train = temp2
		total_x += X_train
		total_y += y_train
	X_train = total_x
	y_train = total_y
	X_test = X_train[-int(test_percent*len(X_train)):]
	y_test = y_train[-int(test_percent*len(y_train)):]
	X_train = X_train[:-int(test_percent*len(X_train))]
	y_train = y_train[:-int(test_percent*len(y_train))]
	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), class_ind

def import_data(file_path, rand, test_percent):
	class_ind = 0
	csvfile = open(file_path,'r')
	data = csv.reader(csvfile)
	X_train = []
	y_train = []
	for row in data:
		class_ind = len(row)-1
		X_train.append([float(x) for x in row[0:class_ind]])
		y_train.append([int(row[class_ind]), int(not(int(row[class_ind])))])
	csvfile.close()
	if rand:
		temp1 = []
		temp2 = []
		index_shuf = list(range(len(X_train)))
		random.shuffle(index_shuf)
		for i in index_shuf:
			temp1.append(X_train[i])
			temp2.append(y_train[i])
		X_train = temp1
		y_train = temp2
	X_test = X_train[-int(test_percent*len(X_train)):]
	y_test = y_train[-int(test_percent*len(y_train)):]
	X_train = X_train[:-int(test_percent*len(X_train))]
	y_train = y_train[:-int(test_percent*len(y_train))]
	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), class_ind


# output = open('KNN_Analysis/KNN_Excerpt_Test.csv', 'w') 
# output.write("K,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc\n")

# clss = 0
# for k in [3,5,7,9,11,13,15,17,19]:
# 	for file1, ind1 in zip(files, [1,2,3,4,5,6,7,8]):
# 		for file2, ind2 in zip(files, [1,2,3,4,5,6,7,8]):
# 			print(str(ind1) + " " +str(ind2))
# 			for i in range(100):
# 				knn = KNeighborsClassifier(n_neighbors=k)

# 				X_train1, y_train1, X_test1, y_test1, clss = import_data(file1,1,.3)
# 				X_train2, y_train2, X_test2, y_test2, clss = import_data(file2,1,.3)

# 				if(file1 == file2):
# 					print((X_train1[0]))
# 					print((y_train1[0]))
# 					print("AAAAAAAAAAAA")
# 					knn.fit(X_train1, y_train1)
# 					pred = knn.predict(X_test1)
# 					acc = accuracy_score(y_test1, pred)
# 					prf = precision_recall_fscore_support(y_test1, pred)
# 					cfm = confusion_matrix([j[0] for j in y_test1], [k[0] for k in pred]).ravel()
# 				else:
# 					print((X_train1[0]))
# 					print((y_train1[0]))
# 					print("bbbbbbbbbbbb")
# 					knn.fit(X_train1, y_train1)
# 					pred = knn.predict(X_test2)
# 					acc = accuracy_score(y_test2, pred)
# 					prf = precision_recall_fscore_support(y_test2, pred)
# 					cfm = confusion_matrix([j[0] for j in y_test2], [j2[0] for j2 in pred]).ravel()

# 				output.write(str(k) + ',' +  str(i) + ',' + str(prf[0][0]) + ',' 
# 					+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
# 					+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
# 					+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
# 					+ str(acc) + '\n')		
# output.close()

output = open('KNN_Analysis/KNN_AllExcerpt_Test_Fmeasure.csv', 'w') 
output.write("K,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc\n")
	
for k in [3,5,7,9,11,13,15,17,19]:
	print(k)
	for i in range(100):
		X_train1, y_train1, X_test1, y_test1, clss = import_all_data(files,1,.3)
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train1, y_train1)
		pred = knn.predict(X_test1)
		acc = accuracy_score(y_test1, pred)
		prf = precision_recall_fscore_support(y_test1, pred)
		cfm = confusion_matrix([j[0] for j in y_test1], [j2[0] for j2 in pred]).ravel()
		output.write(str(k) + ',' +  str(i) + ',' + str(prf[0][0]) + ',' 
			+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
			+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
			+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
			+ str(acc) + '\n')
output.close()

