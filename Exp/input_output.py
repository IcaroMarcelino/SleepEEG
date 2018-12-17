import random
import numpy
import numpy as np
import csv
from deap import tools
import os

def verify_create_dir(path):
	try:
		os.stat(path)
	except:
		os.mkdir(path) 	

	if(path[len(path) - 1] != '/'):
		path += '/'

	try:
		os.stat(path + 'log')
	except:
		os.mkdir(path + 'log') 

	try:
		os.stat(path + 'best_expr')
	except:
		os.mkdir(path + 'best_expr')

	return

def init_stats():
	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_fit.register("avg", numpy.mean, axis=0)
	stats_fit.register("std", numpy.std, axis=0)
	stats_fit.register("min", numpy.min, axis=0)
	stats_fit.register("max", numpy.max, axis=0)

	stats_size = tools.Statistics(lambda ind: ind.height)
	stats_size.register("avg", numpy.mean)
	stats_size.register("std", numpy.std)
	stats_size.register("min", numpy.min)
	stats_size.register("max", numpy.max)

	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	return mstats

def import_all_data(files_paths, rand, test_percent, balance, train_type, op = 0):
	total_x = []
	total_y = []
	class_ind = 0
	for file_path in files_paths:
		csvfile = open(file_path,'r')
		data = csv.reader(csvfile)
		X_total = []
		y_total = []
		X_S = []
		y_S = []
		X_NS = []
		y_NS = []
		X_train = []
		y_train = []
		
		for row in data:
			class_ind = len(row)-2
			X_train.append([float(x) for x in row[0:class_ind]])
			#X_train.append([float(x) for x in row[0:25]])
			y_train.append([int(row[class_ind+op]), int(not(int(row[class_ind+op])))])

			# if row[len(row) - 1] != row[len(row)-2]:
			# 	print(row)
			
			samp = row[0:class_ind] + [row[class_ind+1]]
			# if row[len(row) - 1] != row[len(row)-2]:
			# 	print(samp)
			# 	input()
				
			X_total.append([float(x) for x in samp[0:class_ind]])
			#X_total.append([float(x) for x in samp[0:25]])
			y_total.append([int(samp[class_ind]), int(not(int(samp[class_ind])))])
			
			if int(row[class_ind]):
				X_S.append([float(x) for x in row[0:class_ind]])
				#X_S.append([float(x) for x in row[0:25]])
				y_S.append([int(row[class_ind]), int(not(int(row[class_ind])))])
			else:
				X_NS.append([float(x) for x in row[0:class_ind]])
				#X_NS.append([float(x) for x in row[0:25]])
				y_NS.append([int(row[class_ind]), int(not(int(row[class_ind])))])
		csvfile.close()
		if balance:
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

	if train_type:
		#return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), 25
		return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), class_ind
	else:
		X_total1 = X_total
		for sample in X_total1:
			if sample in X_train:
				ind = X_total.index(sample)
				X_total.remove(X_total[ind])
				y_total.remove(y_total[ind])

		#return np.array(X_total), np.array(y_total), np.array(X_train), np.array(y_train), class_ind
		return np.array(X_train), np.array(y_train), np.array(X_total), np.array(y_total), class_ind
		#return np.array(X_train), np.array(y_train), np.array(X_total), np.array(y_total), 25

def import_data(file_path, rand, test_percent):
	class_ind = 0
	csvfile = open(file_path,'r')
	data = csv.reader(csvfile)
	X_train = []
	y_train = []
	for row in data:
		class_ind = len(row)-2
		X_train.append([float(x) for x in row[0:75]])
		y_train.append([int(row[class_ind+1]), int(not(int(row[class_ind+1])))])
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
