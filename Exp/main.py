from __future__ import absolute_import,division,print_function
import os, shutil

if os.path.isdir("__pycache__"):
	shutil.rmtree("__pycache__")

import operator
import math

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import tools

import time
import sys

from operator_set import*
from fitness_function import*
from input_output import*

def main(NEXEC, K, TAM_MAX, NGEN, CXPB, MUTPB, NPOP, train_percent, verb, FILE_NAME, path, dt_op, opt_vars, wts_vars):
	verify_create_dir(path)

	files_pca 	= ['data/pca_ex1.csv', 			'data/pca_ex2.csv', 		'data/pca_ex3.csv',
					'data/pca_ex4.csv', 		'data/pca_ex5.csv', 		'data/pca_ex6.csv',
					'data/pca_ex7.csv',			'data/pca_ex8.csv']

	files_wav25 = ['data/wav_seg_ex1.csv', 		'data/wav_seg_ex2.csv', 	'data/wav_seg_ex3.csv',
					'data/wav_seg_ex4.csv', 	'data/wav_seg_ex5.csv', 	'data/wav_seg_ex6.csv', 
					'data/wav_seg_ex7.csv', 	'data/wav_seg_ex8.csv']

	files_wav75 = ['data/wav_all_seg_ex1.csv', 	'data/wav_all_seg_ex2.csv', 'data/wav_all_seg_ex3.csv',
					'data/wav_all_seg_ex4.csv', 'data/wav_all_seg_ex5.csv', 'data/wav_all_seg_ex6.csv',
					'data/wav_all_seg_ex7.csv', 'data/wav_all_seg_ex8.csv']
	
	if dt_op == 1:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav25,1, 1-train_percent)
	elif dt_op == 2:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75,1, 1-train_percent)
	elif dt_op == 3:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_pca,1, 1-train_percent)

	eval_func = eval_function(opt_vars)
	########## Operator Set #########################################
	pset = gp.PrimitiveSet("MAIN", n_att)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(plog, 1)
	pset.addPrimitive(psqrt, 1)
	pset.addPrimitive(F, 1)
	pset.addPrimitive(pdiv, 2)
	#pset.addPrimitive(operator.neg, 1)
	#################################################################
	creator.create("FitnessMulti", base.Fitness, weights=wts_vars)
	creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
	toolbox = base.Toolbox()
	toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
	toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("compile", gp.compile, pset=pset)
	toolbox.register("evaluate", eval_tree, K = K, X_train = X_train, y_train = y_train, X_test = X_test, y_true = y_test, pset = pset, toolbox = toolbox, opt_vars = opt_vars, eval_func = eval_func)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("mate", gp.cxOnePoint)
	toolbox.register("expr_mut", gp.genFull, min_=4, max_=7)
	toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
	#toolbox.register("mutate", gp.mutShrink)

	toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))
	toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))

	start = time.time()
	random.seed(318)

	mstats = init_stats()

	pop = toolbox.population(NPOP)
	
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
		if (math.isnan(fit[0])):
			ind.fitness.values = 0,
		else:
			ind.fitness.values = fit

	log = tools.Logbook()
	hof = tools.selBest(pop, 1)

	if verb >= 1:
		print(">> (Exec " + str(NEXEC) + ") GP + KNN - Feature Selection and Classification")
		print(">> NGEN = " + str(NGEN) + " | NPOP = " + str(NPOP) + " | MAX_DEPTH = " + str(TAM_MAX) + " | K = " + str(K))
		print(">> Optimizing: " + str(opt_vars))

	for g in range(NGEN):
		geninit = time.time()

		pop = toolbox.select(pop, NPOP)

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

		if(verb == 2):
			print(log.stream)

	end = time.time()

	total_time = end - start

	if verb >= 1:
		if total_time < 60:
			print(">> End (" + str(round(total_time)) + " seconds)\n")
		elif total_time < 3600:
			print(">> End (" + str(round(total_time/60)) + " minutes)\n")
		else:
			print(">> End (" + str(math.floor(total_time/3600)) + " hours and " + str(round(abs(math.floor(total_time/3600)-total_time/3600)*60)) + " minutes)\n")

	logfile = open(path + "log/LOG_" + filename + "_" + str(NEXEC) + ".csv", 'w')
	logfile.write(str(log))
	logfile.close()

	tree = gp.PrimitiveTree(hof[0])
	expFILE = open(path + "best_expr/EXPR_" + filename + "_" +  str(NEXEC) + ".txt", 'w')
	expFILE.write(str(tree))
		
	info_file_name = path + "infoGP.csv"
	infoGP = open(info_file_name, 'a')
	if os.stat(info_file_name).st_size == 0:
		infoGP.write("DEEP MAX,K,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc,Deep,Training Time\n")
	
	prf, acc, cfm = performance(hof[0], K, X_train, y_train, X_test, y_test, toolbox, pset)

	infoGP.write(str(TAM_MAX) + ',' + str(K) + ',' +  str(NEXEC) + ',' + str(prf[0][0]) + ',' 
				+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
				+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
				+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
				+ str(acc) + ',' + str(hof[0].height) + ',' + str(total_time) + '\n')

	infoGP.close()

if __name__ == "__main__":
	###########################################
	# Default parameters
	###########################################
	NGEN = 100
	NPOP = 20
	tam_max = 10
	K = 5
	execs = [1,2,3,4,5,6,7,8,9,10]
	file_id = "Default_"
	path = "Default_Try/"
	dt_op = 1
	verb = 1

	opt_vars = []
	wts_vars = []

	###########################################
	# User's parameters (If exists)
	###########################################
	for i in range(len(sys.argv)-1):  
		if (sys.argv[i] == '-gen'):
			NGEN = int(sys.argv[i+1])

		elif(sys.argv[i] == '-pop'):
			NPOP = int(sys.argv[i+1])

		elif(sys.argv[i] == '-depth'):
			tam_max = int(sys.argv[i+1])

		elif(sys.argv[i] == '-k'):
			K = int(sys.argv[i+1])

		elif(sys.argv[i] == '-execs'):
			RANG_EXEC_1 = int(sys.argv[i+1])
			RANG_EXEC_2 = int(sys.argv[i+2])+1
			execs = list(range(RANG_EXEC_1, RANG_EXEC_2))

		elif(sys.argv[i] == '-dataset'):
			dt_op = int(sys.argv[i+1])

		elif(sys.argv[i] == '-path'):
			path = sys.argv[i+1]

		elif(sys.argv[i] == '-fileID'):
			file_id = sys.argv[i+1]

		elif(sys.argv[i] == '-optmize'):
			n_vars = int(sys.argv[i+1])

			j = i+2
			while j <= (2*n_vars + i+1):
				wts_vars.append(int(sys.argv[j]))
				opt_vars.append(sys.argv[j+1])
				j += 2

			wts_vars = tuple(wts_vars)

		elif(sys.argv[i] == '-v'):
			verb = int(sys.argv[i+1])

	CXPB = .8
	MUTPB = .2
	train_percent = 0.7
	
	filename = file_id + "GP_EEG_K" + str(K) + "_"	
	
	if len(opt_vars) == 0:
		opt_vars = ['acc']
		wts_vars = tuple([1])

	# print(NGEN)
	# print(NPOP)
	# print(tam_max)
	# print(K)
	# print(execs)
	# print(dt_op)
	# print(path)
	# print(file_id)
	# print(opt_vars)
	# print(wts_vars)

	for i in execs:
		main(	NEXEC = i,
				K = K,
				TAM_MAX = tam_max,
				NGEN = NGEN,
				CXPB = CXPB, 	
				MUTPB = MUTPB,
				NPOP = NPOP,
				train_percent = train_percent, 
				verb = verb, 
				FILE_NAME = filename,
				path = path,
				dt_op = dt_op,
				opt_vars = opt_vars,
				wts_vars = wts_vars)
