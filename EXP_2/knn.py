from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
import csv
import random
import numpy as np


files = ['wav_seg_ex1.csv', 'wav_seg_ex2.csv', 'wav_seg_ex3.csv', 'wav_seg_ex4.csv', 'wav_seg_ex5.csv', 'wav_seg_ex6.csv', 'wav_seg_ex7.csv', 'wav_seg_ex8.csv']
files_men = ['wav_seg_ex2.csv', 'wav_seg_ex3.csv', 'wav_seg_ex4.csv', 'wav_seg_ex8.csv']
files_wom = ['wav_seg_ex1.csv', 'wav_seg_ex5.csv', 'wav_seg_ex6.csv', 'wav_seg_ex7.csv']

def import_data(file_path, rand, test_percent):
	csvfile = open(file_path,'r')
	data = csv.reader(csvfile)
	X_train = []
	y_train = []
	for row in data:
		X_train.append([float(x) for x in row[0:25]])
		y_train.append([int(row[25]), int(not(int(row[25])))])
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
	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def import_all_data(files_paths, rand, test_percent):
	total_x = []
	total_y = []
	for file_path in files_paths:
		csvfile = open(file_path,'r')
		data = csv.reader(csvfile)
		X_train = []
		y_train = []
		for row in data:
			X_train.append([float(x) for x in row[0:25]])
			y_train.append([int(row[25]), int(not(int(row[25])))])
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
		total_x += X_train
		total_y += y_train
	X_train = total_x
	y_train = total_y
	X_test = X_train[-int(test_percent*len(X_train)):]
	y_test = y_train[-int(test_percent*len(y_train)):]
	X_train = X_train[:-int(test_percent*len(X_train))]
	y_train = y_train[:-int(test_percent*len(y_train))]
	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


output = open('KNN_Analysis/KNN_Excerpt_Test.csv', 'w') 
output.write('K, Exec, Excerpt_Train, Excerpt_Test, Acc\n')

for k in [3,5,7,9,11,13,15,17,19]:
	out_ave = open('KNN_Analysis/KNN_AVE_K' + str(k) + '.csv', 'w')
	out_std = open('KNN_Analysis/KNN_STD_K' + str(k) + '.csv', 'w')

	for file1, ind1 in zip(files, [1,2,3,4,5,6,7,8]):
		excerpt_results = []
		for file2, ind2 in zip(files, [1,2,3,4,5,6,7,8]):
			print(ind1, " ", ind2)
			result = []
			for i in range(100):
				X_train1, y_train1, X_test1, y_test1 = import_data(file1,1,.3)
				X_train2, y_train2, X_test2, y_test2 = import_data(file2,1,.3)

				knn = KNeighborsClassifier(n_neighbors=k)

				if(file1 == file2):
					knn.fit(X_train1, y_train1)
					pred = knn.predict(X_test1)
					acc = accuracy_score(y_test1, pred)
				else:
					knn.fit(X_train1, y_train1)
					pred = knn.predict(X_test2)
					acc = accuracy_score(y_test2, pred)

				result.append(acc)
				output.write(str(k) + ',' + str(i) + ',' +  str(ind1) + ',' + str(ind2) + ',' + str(acc) + '\n')
			excerpt_results.append([np.mean(result), np.std(result)])

		out_ave.write(str(excerpt_results[0][0]) + ',' + str(excerpt_results[1][0]) + ',' + str(excerpt_results[2][0]) + ',' + str(excerpt_results[3][0]) + ',' + str(excerpt_results[4][0]) + ',' + str(excerpt_results[5][0]) + ',' + str(excerpt_results[6][0]) + ',' + str(excerpt_results[7][0]) + '\n')
		out_std.write(str(excerpt_results[0][1]) + ',' + str(excerpt_results[1][1]) + ',' + str(excerpt_results[2][1]) + ',' + str(excerpt_results[3][1]) + ',' + str(excerpt_results[4][1]) + ',' + str(excerpt_results[5][1]) + ',' + str(excerpt_results[6][1]) + ',' + str(excerpt_results[7][1]) + '\n')
	out_ave.close()
	out_std.close()

output.close()


output = open('KNN_Analysis/KNN_AllExcerpt_Test.csv', 'w') 
output.write('K, Exec, Acc\n')

for k in [3,5,7,9,11,13,15,17,19]:
	result = []
	for i in range(100):
		X_train1, y_train1, X_test1, y_test1 = import_all_data(files_paths,1,.3)
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train1, y_train1)
		pred = knn.predict(X_test1)
		acc = accuracy_score(y_test1, pred)
		result.append(acc)
		output.write(str(k) + ',' + str(i) + ',' + str(acc) + '\n')
output.close()

