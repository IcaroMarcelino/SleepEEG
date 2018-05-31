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
#import pygraphviz as pgv



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
			return 0,
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
			return 0,
		X_test_new = []
		i = 0
		for x in X_test:
			X_test_new.append([])
			for feature in features:
				X_test_new[i].append(feature(*x))
			i += 1
		# predict the response
		pred = knn.predict(X_test_new)
		# evaluate accuracy
		#print accuracy_score(y_test, pred)
		return accuracy_score(y_test, pred),

	X_train, y_train, X_test, y_test = import_data('wav_seg_ex1.csv',1,.3)

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
	creator.create("FitnessMin", base.Fitness, weights=(1.0,))
	creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
	toolbox = base.Toolbox()
	toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)
	toolbox.register("evaluate", eval_tree, K = K, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, pset = pset)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))

	start = time.time()
	random.seed(318)

	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(lambda ind: ind.height)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", numpy.mean)
	mstats.register("std", numpy.std)
	mstats.register("min", numpy.min)
	mstats.register("max", numpy.max)

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
	if (NEXEC == 0):
		info.write("Altura Maxima,K,#Execucao,Acc (Melhor),Altura (Melhor),Tempo Execucao\n")

	info.write(str(TAM_MAX) + ',' + str(K) + ',' +  str(NEXEC + 1) + ',' + str(toolbox.evaluate(hof[0])[0]) + ',' + str(hof[0].height) + ',' + str(end-start) + '\n')

	info1 = open("INFO_GP.csv", 'a')
	info1.write(str(TAM_MAX) + ',' + str(K) + ',' +  str(NEXEC + 1) + ',' + str(toolbox.evaluate(hof[0])[0]) + ',' + str(hof[0].height) + ',' + str(end-start) + '\n')

	# nodes, edges, labels = gp.graph(hof[0])

	# g = pgv.AGraph()
	# g.add_nodes_from(nodes)
	# g.add_edges_from(edges)
	# g.layout(prog="dot")

	# for i in nodes:
	# 	n = g.get_node(i)
	# 	n.attr["label"] = labels[i]

	# g.draw("Grafos_Melhores/GRAPH_" + filename +  "_" + str(NEXEC + 1) + ".pdf")
	hof = []

if __name__ == "__main__":
	NGEN = 300
	CXPB = .8
	MUTPB = .2
	NPOP = 500
	train_percent = 0.7
	tam_max = 20
	Ks = [13]


	for K in Ks:
		filename = "GP_EEG_K" + str(K) + "_"
	
		for i in [6,7,8,9,10]:
			main(	NEXEC = i,
					K = K,
					TAM_MAX = tam_max,
					NGEN = NGEN,
					CXPB = CXPB, 	
					MUTPB = MUTPB,
					NPOP = NPOP,
					train_percent = train_percent, 
					verb = True, 
					FILE_NAME = filename)