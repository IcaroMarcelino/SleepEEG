from __future__ import absolute_import,division,print_function

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import numpy as np
# import pygraphviz as pgv
# from sympy import simplify, expand
import time
import csv

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support

#import pygraphviz as pgv

import sys

def import_all_data(files_paths, rand, test_percent):
	total_x = []
	total_y = []
	for file_path in files_paths:
		csvfile = open(file_path,'r')
		data = csv.reader(csvfile)
		X_S = []
		y_S = []
		X_NS = []
		y_NS = []
		for row in data:
			if int(row[25]):
				X_S.append([float(x) for x in row[0:25]])
				y_S.append([int(row[25]), int(not(int(row[25])))])
			else:
				X_NS.append([float(x) for x in row[0:25]])
				y_NS.append([int(row[25]), int(not(int(row[25])))])
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
	return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

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


def main(NEXEC, K, TAM_MAX, NGEN, CXPB, MUTPB, NPOP, train_percent, verb, FILE_NAME):
	def div(left, right):
		try:
			return left / right
		except ZeroDivisionError:
			return 1

	def plog(x):
		try:
			return math.log(abs(x))
		except:
			return 1

	def F(x):
		return(x)

	def psqrt(x):
		return abs(x)**(.5)

	def get_subtree(begin, string):
		parentesis = 0
		end = begin
		flag = 0
		for char in string[begin:len(string)]:
			if char == '(':
				flag = 1
				parentesis += 1
			elif char == ')':
				parentesis -= 1
			end += 1
			if parentesis == 0 and flag == 1:
				break
		return string[begin:end]

	def eval_tree(individual, K, X_train, y_train, X_test, y_test, pset):
		# Transform the tree expression in a callable function
		# f_approx = toolbox.compile(expr=individual)
		# Evaluate the mean squared error between the expression and the real function
		# sqerrors = ((f_approx(x,y) - f(x,y))**2 for x,y in points)
		exp = gp.PrimitiveTree(individual)
		string = str(exp)
		ind = [i for i in range(len(string)) if string.startswith('F', i)]
		features = []
		for i in ind:
			subtree = get_subtree(i,string)
			newtree = exp.from_string(subtree, pset)
			features.append(toolbox.compile(newtree))
		if len(features) == 0:
			features.append(toolbox.compile(individual))
		X_train_new = []
		i = 0
		for x in X_train:
			X_train_new.append([])
			for feature in features:
				X_train_new[i].append(feature(*x))
			i += 1
		# instantiate learning model (k = 3)
		knn = KNeighborsClassifier(n_neighbors=K)
		# fitting the model
		try:
			knn.fit(X_train_new, y_train)
		except:
			return 0, 0, 0
		X_test_new = []
		i = 0
		for x in X_test:
			X_test_new.append([])
			for feature in features:
				X_test_new[i].append(feature(*x))
			i += 1
		# predict the response
		pred = knn.predict(X_test_new)
		acc1 = precision_recall_fscore_support(y_test, pred)
		# evaluate accuracy
		#print accuracy_score(y_test, pred)

		return acc1[0][0], acc1[1][0], accuracy_score(y_test, pred)

	def eval_tree1(individual, K, X_train, y_train, X_test, y_test, pset):
		# Transform the tree expression in a callable function
		# f_approx = toolbox.compile(expr=individual)
		# Evaluate the mean squared error between the expression and the real function
		# sqerrors = ((f_approx(x,y) - f(x,y))**2 for x,y in points)
		exp = gp.PrimitiveTree(individual)
		string = str(exp)
		ind = [i for i in range(len(string)) if string.startswith('F', i)]
		features = []
		for i in ind:
			subtree = get_subtree(i,string)
			newtree = exp.from_string(subtree, pset)
			features.append(toolbox.compile(newtree))
		if len(features) == 0:
			features.append(toolbox.compile(individual))
		X_train_new = []
		i = 0
		for x in X_train:
			X_train_new.append([])
			for feature in features:
				X_train_new[i].append(feature(*x))
			i += 1
		# instantiate learning model (k = 3)
		knn = KNeighborsClassifier(n_neighbors=K)
		# fitting the model
		try:
			knn.fit(X_train_new, y_train)
		except:
			return 0, 0, 0
		X_test_new = []
		i = 0
		for x in X_test:
			X_test_new.append([])
			for feature in features:
				X_test_new[i].append(feature(*x))
			i += 1
		# predict the response
		pred = knn.predict(X_test_new)
		acc1 = precision_recall_fscore_support(y_test, pred)
		# evaluate accuracy
		#print accuracy_score(y_test, pred)
		return acc1

	files = ['wav_seg_ex1.csv', 'wav_seg_ex2.csv', 'wav_seg_ex3.csv', 'wav_seg_ex4.csv', 'wav_seg_ex5.csv', 'wav_seg_ex6.csv', 'wav_seg_ex7.csv', 'wav_seg_ex8.csv']
	X_train, y_train, X_test, y_test = import_all_data(files,1,.3)

	########## Operator Set #########################################
	pset = gp.PrimitiveSet("MAIN", 25)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(plog, 1)
	pset.addPrimitive(psqrt, 1)
	pset.addPrimitive(F, 1)
	pset.addPrimitive(div, 2)
	#pset.addPrimitive(operator.neg, 1)
	#################################################################
	creator.create("FitnessMin", base.Fitness, weights=(1.0,1.0,1.0))
	creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
	toolbox = base.Toolbox()
	toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)
	toolbox.register("evaluate", eval_tree, K = K, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, pset = pset)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genFull, min_=4, max_=7)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
	#toolbox.register("mutate", gp.mutShrink)

	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))

	start = time.time()
	random.seed(318)

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
	
	pop = toolbox.population(NPOP)
	
	# pop, log = algorithms.eaSimple(population = pop, toolbox = toolbox, cxpb = CXPB, mutpb = MUTPB, ngen = NGEN, stats = mstats,
	# 							   halloffame = hof, verbose = True)
	
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
		if (math.isnan(fit[0])):
			ind.fitness.values = 0,
		else:
			ind.fitness.values = fit

	log = tools.Logbook()
	hof = tools.selBest(pop, 1)

	print(">> GP Feature Selection: ")
	print(">> Inicio (Execucao " + str(NEXEC) +")")
	for g in range(NGEN):
		geninit = time.time()
		pop = toolbox.select(pop, len(pop))

		offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

		fitnesses = list(map(toolbox.evaluate, offspring))
		for ind, fit in zip(offspring, fitnesses):
			ind.fitness.values = fit
			if (math.isnan(fit[0])):
				ind.fitness.values = 0,
			else:
				ind.fitness.values = fit

		hof = tools.selBest(pop, 1)
		pop[:] = offspring + hof

		#print(toolbox.evaluate(hof[0]))

		log.record(gen = g, time = time.time() - geninit,**mstats.compile(pop))

		if(verb == True):
			print(log.stream)

	end = time.time()


	logfile = open("Log_Exec/LOG_" + filename + "_" + str(NEXEC + 1) + ".csv", 'w')
	logfile.write(str(log))
	print(">> Fim da execucao (" + str(end - start) + " segundos)\n")


	# final_eq = expand(simplify(convertFunct(hof[0])))
	# print("Resultado da regressao: %s\n" % final_eq)


	tree = gp.PrimitiveTree(hof[0])
	function = gp.compile(tree, pset)
	expFILE = open("Grafos_Melhores/EXPR_" + filename + "_" +  str(NEXEC + 1) + ".txt", 'w')
	expFILE.write(str(tree))
	

	info = open("Info/INFO_" + filename + ".csv", 'a')
	# if (NEXEC == 0):
	# 	info.write("Altura Maxima,K,#Execucao,Precision_S,Precision_NS,Recall_S,Recall_NS,Fscore_S,Fscore_NS,Support_S,Support_NS,Acc (Melhor),Altura (Melhor),Tempo Execucao\n")

	info.write(str(TAM_MAX) + ',' + str(K) + ',' +  str(NEXEC + 1) + ',' + str(toolbox.evaluate(hof[0])[2]) + ',' + str(hof[0].height) + ',' + str(end-start) + '\n')

	acc1 = eval_tree1(hof[0], K, X_train, y_train, X_test, y_test, pset)
	info1 = open("INFO_GP.csv", 'a')
	info1.write(str(TAM_MAX) + ',' + str(K) + ',' +  str(NEXEC + 1) + ',' + str(toolbox.evaluate(hof[0])[2]) + ',' + str(acc1[0][0]) + ',' + str(acc1[0][1]) + ',' + str(acc1[1][0])  + ',' + str(acc1[1][1]) + ',' + str(acc1[2][0]) + ',' + str(acc1[2][1]) + ',' + str(acc1[3][0]) + ',' + str(acc1[3][1]) + ','  + str(hof[0].height) + ',' + str(end-start) + '\n')

	# nodes, edges, labels = gp.graph(hof[0])

	# g = pgv.AGraph()
	# g.add_nodes_from(nodes)
	# g.add_edges_from(edges)
	# g.layout(prog="dot")

	# for i in nodes:
	# 	n = g.get_node(i)
	# 	n.attr["label"] = labels[i]

	# g.draw("Grafos_Melhores/GRAPH_" + filename +  "_" + str(NEXEC + 1) + ".pdf")
	# hof = []

if __name__ == "__main__":
	# 300, 300, 20, K, execs
	NGEN = int(sys.argv[1])
	NPOP = int(sys.argv[2])
	CXPB = .8
	MUTPB = .2
	train_percent = 0.7
	tam_max = int(sys.argv[3])
	K = int(sys.argv[4])

	if sys.argv[5] == 0:
		execs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
	else:
		execs = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

	filename = "GP_EEG_K" + str(K) + "_"
	
	for i in execs:
		main(	NEXEC = i,
				K = K,
				TAM_MAX = tam_max,
				NGEN = NGEN,
				CXPB = CXPB, 	
				MUTPB = MUTPB,
				NPOP = NPOP,
				train_percent = train_percent, 
				verb = False, 
				FILE_NAME = filename)