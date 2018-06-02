from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


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

	X_train, y_train, X_test, y_test = import_data('wav_seg_ex1.csv',1,.3)