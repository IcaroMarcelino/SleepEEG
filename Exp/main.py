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

import random
import time
import sys

from operator_set import*
from fitness_function import performance, eval_function, eval_tree
from input_output import import_all_data, import_data, verify_create_dir, init_stats

def main(NEXEC, classifier, clf_param, TAM_MAX, NGEN, CXPB, MUTPB, NPOP, train_percent, verb, FILE_NAME, path, dt_op, opt_vars, wts_vars, ini, sel, mut, crs, balance, train_type):
	verify_create_dir(path)

	files_pca 	= ['data/pca_ex1.csv', 			'data/pca_ex2.csv', 		'data/pca_ex3.csv',
					'data/pca_ex4.csv', 		'data/pca_ex5.csv', 		'data/pca_ex6.csv',
					'data/pca_ex7.csv',			'data/pca_ex8.csv']

	files_wav25 = [	'data/data_25/wav25_ex1_.csv', 'data/data_25/wav25_ex2_.csv', 'data/data_25/wav25_ex3_.csv',
					'data/data_25/wav25_ex4_.csv', 'data/data_25/wav25_ex5_.csv', 'data/data_25/wav25_ex6_.csv',
					'data/data_25/wav25_ex7_.csv', 'data/data_25/wav25_ex8_.csv']

	files_wav75 = [	'data/data_75/wav75_ex1_.csv', 'data/data_75/wav75_ex2_.csv', 'data/data_75/wav75_ex3_.csv',
					'data/data_75/wav75_ex4_.csv', 'data/data_75/wav75_ex5_.csv', 'data/data_75/wav75_ex6_.csv',
					'data/data_75/wav75_ex7_.csv', 'data/data_75/wav75_ex8_.csv']

	files_wav75_rms = ['data/wav1_all_seg_ex1_RMS.csv', 	'data/wav1_all_seg_ex2_RMS.csv', 'data/wav1_all_seg_ex3_RMS.csv',
					'data/wav1_all_seg_ex4_RMS.csv', 'data/wav1_all_seg_ex5_RMS.csv', 'data/wav1_all_seg_ex6_RMS.csv',
					'data/wav1_all_seg_ex7_RMS.csv', 'data/wav1_all_seg_ex8_RMS.csv']
	
	files_wav75_filter = ['data/wav1_all_seg_ex1_S05_W1_F.csv', 	'data/wav1_all_seg_ex2_S05_W1_F.csv', 'data/wav1_all_seg_ex3_S05_W1_F.csv',
					'data/wav1_all_seg_ex4_S05_W1_F.csv', 'data/wav1_all_seg_ex5_S05_W1_F.csv', 'data/wav1_all_seg_ex6_S05_W1_F.csv',
					'data/wav1_all_seg_ex7_S05_W1_F.csv', 'data/wav1_all_seg_ex8_S05_W1_F.csv']

	files_wav75_men = [	'data/data_75/wav75_ex2_.csv', 'data/data_75/wav75_ex3_.csv',
						'data/data_75/wav75_ex4_.csv', 'data/data_75/wav75_ex8_.csv']

	files_wav75_wom = [	'data/data_75/wav75_ex1_.csv', 'data/data_75/wav75_ex5_.csv', 
						'data/data_75/wav75_ex6_.csv', 'data/data_75/wav75_ex7_.csv']

	files_wav75_exp1 = ['data/wav1_all_seg_ex1_exp1.csv', 	'data/wav1_all_seg_ex2_exp1.csv', 'data/wav1_all_seg_ex3_exp1.csv',
					'data/wav1_all_seg_ex4_exp1.csv', 'data/wav1_all_seg_ex5_exp1.csv', 'data/wav1_all_seg_ex6_exp1.csv',
					'data/wav1_all_seg_ex7_exp1.csv', 'data/wav1_all_seg_ex8_exp1.csv']

	files_wav75_exp2 = ['data/wav1_all_seg_ex1_exp2.csv', 	'data/wav1_all_seg_ex2_exp2.csv', 'data/wav1_all_seg_ex3_exp2.csv',
					'data/wav1_all_seg_ex4_exp2.csv', 'data/wav1_all_seg_ex5_exp2.csv', 'data/wav1_all_seg_ex6_exp2.csv']

	files_wav75_01=['data/data_75/wav75_ex1_01.csv', 'data/data_75/wav75_ex2_01.csv', 'data/data_75/wav75_ex3_01.csv',
					'data/data_75/wav75_ex4_01.csv', 'data/data_75/wav75_ex5_01.csv', 'data/data_75/wav75_ex6_01.csv',
					'data/data_75/wav75_ex7_01.csv', 'data/data_75/wav75_ex8_01.csv']
	
	files_wav75_00=['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	files_wav75_FF =['data/data_75/wav_ex1_Filtered.csv', 'data/data_75/wav_ex2_Filtered.csv', 'data/data_75/wav_ex3_Filtered.csv',
					'data/data_75/wav_ex4_Filtered.csv', 'data/data_75/wav_ex5_Filtered.csv', 'data/data_75/wav_ex6_Filtered.csv',
					'data/data_75/wav_ex7_Filtered.csv', 'data/data_75/wav_ex8_Filtered.csv']

	files_wav75_FN =['data/data_75/wav_ex1_Filtered_Norm_STP.csv', 'data/data_75/wav_ex2_Filtered_Norm_STP.csv', 'data/data_75/wav_ex3_Filtered_Norm_STP.csv',
					'data/data_75/wav_ex4_Filtered_Norm_STP.csv', 'data/data_75/wav_ex5_Filtered_Norm_STP.csv', 'data/data_75/wav_ex6_Filtered_Norm_STP.csv',
					'data/data_75/wav_ex7_Filtered_Norm_STP.csv', 'data/data_75/wav_ex8_Filtered_Norm_STP.csv']

	files_wav75_FNN =['data/data_75/wav_ex1_Filtered_N_Norm_STP.csv', 'data/data_75/wav_ex2_Filtered_N_Norm_STP.csv', 'data/data_75/wav_ex3_Filtered_N_Norm_STP.csv',
					'data/data_75/wav_ex4_Filtered_N_Norm_STP.csv', 'data/data_75/wav_ex5_Filtered_N_Norm_STP.csv', 'data/data_75/wav_ex6_Filtered_N_Norm_STP.csv',
					'data/data_75/wav_ex7_Filtered_N_Norm_STP.csv', 'data/data_75/wav_ex8_Filtered_N_Norm_STP.csv']

	files_wav75_NormF =['data/data_75/wav_ex1_Norm.csv', 'data/data_75/wav_ex2_Norm.csv', 'data/data_75/wav_ex3_Norm.csv',
					'data/data_75/wav_ex4_Norm.csv', 'data/data_75/wav_ex5_Norm.csv', 'data/data_75/wav_ex6_Norm.csv',
					'data/data_75/wav_ex7_Norm.csv', 'data/data_75/wav_ex8_Norm.csv']
	
	files_wav75_F11 =['data/data_75/wav_ex1_Filtered_11F_N_Norm.csv', 'data/data_75/wav_ex2_Filtered_11F_N_Norm.csv', 'data/data_75/wav_ex3_Filtered_11F_N_Norm.csv',
					'data/data_75/wav_ex4_Filtered_11F_N_Norm.csv', 'data/data_75/wav_ex5_Filtered_11F_N_Norm.csv', 'data/data_75/wav_ex6_Filtered_11F_N_Norm.csv',
					'data/data_75/wav_ex7_Filtered_11F_N_Norm.csv', 'data/data_75/wav_ex8_Filtered_11F_N_Norm.csv']

	files_wav75_KC = [	'data/KC/data_75/wav_ex1_.csv', 'data/KC/data_75/wav_ex2_.csv', 'data/KC/data_75/wav_ex3_.csv',
					'data/KC/data_75/wav_ex4_.csv', 'data/KC/data_75/wav_ex5_.csv', 'data/KC/data_75/wav_ex6_.csv',
					'data/KC/data_75/wav_ex7_.csv', 'data/KC/data_75/wav_ex8_.csv']
	
	files_wav75_KC_men = [	'data/KC/data_75/wav_ex1_.csv', 'data/KC/data_75/wav_ex6_.csv', 'data/KC/data_75/wav_ex7_.csv', 'data/KC/data_75/wav_ex9_.csv']

	files_wav75_KC_wom = ['data/KC/data_75/wav_ex2_.csv', 'data/KC/data_75/wav_ex3_.csv',
					'data/KC/data_75/wav_ex4_.csv', 'data/KC/data_75/wav_ex5_.csv',
					'data/KC/data_75/wav_ex10_.csv']

	files_PCA = ['data/pca1_ex0_ALL.csv']

	kf1 =['data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	kf2 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	kf3 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	kf4 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	kf5 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	kf6 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv',
					'data/data_75/wav75_ex7_0.csv', 'data/data_75/wav75_ex8_0.csv']

	kf7 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex8_0.csv']

	kf8 =['data/data_75/wav75_ex1_0.csv', 'data/data_75/wav75_ex2_0.csv', 'data/data_75/wav75_ex3_0.csv',
					'data/data_75/wav75_ex4_0.csv', 'data/data_75/wav75_ex5_0.csv', 'data/data_75/wav75_ex6_0.csv',
					'data/data_75/wav75_ex7_0.csv']


	if dt_op == 1:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav25,1, 1-train_percent, balance, train_type)
	elif dt_op == 2:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75,1, 1-train_percent, balance, train_type)
	elif dt_op == 3:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_pca,1, 1-train_percent, balance, train_type)
	elif dt_op == 4:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_men,1, 1-train_percent, balance, train_type)
	elif dt_op == 5:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_wom,1, 1-train_percent, balance, train_type)
	elif dt_op == 6:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_exp1,1, 1-train_percent, balance, train_type)
	elif dt_op == 7:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_exp2,1, 1-train_percent, balance, train_type)
	elif dt_op == 8:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_filter,1, 1-train_percent, balance, train_type)
	elif dt_op == 9:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_rms,1, 1-train_percent, balance, train_type)
	elif dt_op == 10:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_01,1, 1-train_percent, balance, train_type)
	elif dt_op == 11:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_00,1, 1-train_percent, balance, train_type)
	elif dt_op == 12:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_FF,1, 1-train_percent, balance, train_type)
	elif dt_op == 13:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_FN,1, 1-train_percent, balance, train_type)
	elif dt_op == 14:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_FNN,1, 1-train_percent, balance, train_type)
	elif dt_op == 15:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_NormF,1, 1-train_percent, balance, train_type)
	elif dt_op == 16:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_F11,1, 1-train_percent, balance, train_type)
	elif dt_op == 17:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_KC,1, 1-train_percent, balance, train_type)
	elif dt_op == 18:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_KC_men,1, 1-train_percent, balance, train_type)
	elif dt_op == 19:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_wav75_KC_wom,1, 1-train_percent, balance, train_type)
	elif dt_op == 19:
		X_train, y_train, X_test, y_test, n_att = import_all_data(files_PCA,1, 1-train_percent, balance, train_type)
	elif dt_op == 21:
		_, _, X_train, y_train, n_att = import_all_data(kf1,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex1_0.csv'],1, 1, 0, 1)
		#print(X_train.shape)
		#print(y_train.shape)
		#print(X_test.shape)
		#print(y_test.shape)
	elif dt_op == 22:
		_, _, X_train, y_train, n_att = import_all_data(kf2,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex2_0.csv'],1, 1, 0, 1)
	elif dt_op == 23:
		_, _, X_train, y_train, n_att = import_all_data(kf3,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex3_0.csv'],1, 1, 0, 1)
	elif dt_op == 24:
		_, _, X_train, y_train, n_att = import_all_data(kf4,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex4_0.csv'],1, 1, 0, 1)
	elif dt_op == 25:
		_, _, X_train, y_train, n_att = import_all_data(kf5,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex5_0.csv'],1, 1, 0, 1)
	elif dt_op == 26:
		_, _, X_train, y_train, n_att = import_all_data(kf6,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex6_0.csv'],1, 1, 0, 1)
	elif dt_op == 27:
		_, _, X_train, y_train, n_att = import_all_data(kf7,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex7_0.csv'],1, 1, 0, 1)
	elif dt_op == 28:
		_, _, X_train, y_train, n_att = import_all_data(kf8,1, 1, 0, 1)
		_, _, X_test, y_test, n_att = import_all_data(['data/data_75/wav75_ex8_0.csv'],1, 1, 0, 1)

	eval_func = eval_function(opt_vars)
	########## Operator Set #########################################
	pset = gp.PrimitiveSet("MAIN", n_att)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(plog, 1)
	#pset.addPrimitive(math.sin, 1)
	#pset.addPrimitive(math.cos, 1)
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
	toolbox.register("evaluate", eval_tree, clf = classifier, param = clf_param, X_train = X_train, y_train = y_train, X_test = X_test, y_true = y_test, pset = pset, opt_vars = opt_vars, eval_func = eval_func)

	################## HYPERPARAMETER ####################################
	################## INITIALIZATION ####################################
	if ini == 1:
		toolbox.register("expr_mut", gp.genFull, min_=4, max_=7)
	elif ini == 2:
		toolbox.register("expr_mut", gp.genGrow, min_=4, max_=7)
	elif ini == 3:
		toolbox.register("expr_mut", gp.HalfAndHalf, min_=4, max_=7)
	
	################## HYPERPARAMETER ####################################
	################## SELECTION      ####################################
	if sel == 1:
		toolbox.register("select", tools.selTournament, tournsize=3)
	elif sel == 2:
		toolbox.register("select", tools.selRoulette)
	elif sel == 3:
		toolbox.register("select", tools.selRandom)
	
	################## HYPERPARAMETER ####################################
	################## CROSSOVER      ####################################
	if crs == 1:
		toolbox.register("mate", gp.cxOnePoint)
	elif crs == 2:
		toolbox.register("mate", gp.cxTwoPoint)
	elif crs == 3:
		toolbox.register("mate", gp.cxcxOnePointLeafBiased, termpb =.1)

	################## HYPERPARAMETER ####################################
	################## MUTATION       ####################################
	if mut == 1:
		toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
	elif mut == 2:
		toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
	elif mut == 3:
		toolbox.register("mutate", gp.mutInsert, pset=pset)
	elif mut == 4:
		toolbox.register("mutate", gp.mutShrink)
	elif mut == 5:
		toolbox.register("mutate", gp.mutEphemeral, mode = 'all')

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

	toolbar_width = 50
	if NGEN < 50:
		toolbar_width = NGEN

	if verb >= 1:
		print(">> (Exec " + str(NEXEC) + ") GP + " + classifier + " - Feature Selection and Classification")
		print(">> NGEN = " + str(NGEN) + " | NPOP = " + str(NPOP) + " | MAX_DEPTH = " + str(TAM_MAX) + " | PARAM = " + str(clf_param))
		print(">> Optimizing: " + str(opt_vars))
		print(">> Weights:    " + str(wts_vars))
		if verb == 1:
			sys.stdout.write("[%s]" % (" " * toolbar_width))
			sys.stdout.flush()
			sys.stdout.write("\b" * (toolbar_width+1))

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

		if verb == 1:
			if(g%int(NGEN/toolbar_width) == 0):
				sys.stdout.write("-")
				sys.stdout.flush()
		elif verb == 2:
			print(log.stream)

	end = time.time()

	total_time = end - start

	if verb >= 1:
		if total_time < 60:
			print("\n>> End (" + str(round(total_time)) + " seconds)\n")
		elif total_time < 3600:
			print("\n>> End (" + str(round(total_time/60)) + " minutes)\n")
		else:
			print("\n>> End (" + str(math.floor(total_time/3600)) + " hours and " + str(round(abs(math.floor(total_time/3600)-total_time/3600)*60)) + " minutes)\n")

	logfile = open(path + "log/LOG_" + filename + "_" + str(NEXEC) + ".csv", 'w')
	logfile.write(str(log))
	logfile.close()

	tree = gp.PrimitiveTree(hof[0])
	expFILE = open(path + "best_expr/EXPR_" + filename + "_" +  str(NEXEC) + ".txt", 'w')
	expFILE.write(str(tree))
		
	prf, acc, cfm, AUC = performance(hof[0], classifier, clf_param, X_train, y_train, X_test, y_test, pset)

	info_file_name = path + "infoGP.csv"
	infoGP = open(info_file_name, 'a')
	if os.stat(info_file_name).st_size == 0:
		infoGP.write("balance,DEEP MAX,classifier,P1,P2,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc,AUC,Deep,Training Time\n")

	infoGP.write(str(balance) + ',' + str(TAM_MAX) + ',' + classifier + ',' + str(clf_param[0]) + ',' + str(clf_param[1]) + ',' +  str(NEXEC) + ',' + str(prf[0][0]) + ',' 
			+ str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
			+ str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
			+ str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
			+ str(acc) + ',' + str(AUC) + ',' + str(hof[0].height) + ',' + str(total_time) + '\n')

	infoGP.close()

if __name__ == "__main__":
	###########################################
	# Default parameters
	###########################################
	NGEN = 10
	NPOP = 20
	tam_max = 5
	clf = 'knn'
	param = 5
	param2 = '-'
	execs = [1]
	file_id = "Default_"
	path = "Default_Try/"
	dt_op = 2
	verb = 1
	ini = 1
	sel = 1
	mut = 1
	crs = 1
	train_type = 1
	balance = 0
	kfold = 0

	opt_vars = []
	wts_vars = []

	###########################################
	# User's parameters (If exists) PYTHON FIRE
	###########################################
	for i in range(len(sys.argv)-1):  
		if (sys.argv[i] == '-gen'):
			NGEN = int(sys.argv[i+1])

		elif(sys.argv[i] == '-pop'):
			NPOP = int(sys.argv[i+1])

		elif(sys.argv[i] == '-depth'):
			tam_max = int(sys.argv[i+1])

		elif(sys.argv[i] == '-param'):
			param = int(sys.argv[i+1])

		elif(sys.argv[i] == '-param2'):
			param2 = sys.argv[i+1]

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

		elif(sys.argv[i] == '-clf'):
			clf = sys.argv[i+1]

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

		elif(sys.argv[i] == '-ini'):
			ini = int(sys.argv[i+1])

		elif(sys.argv[i] == '-sel'):
			sel = int(sys.argv[i+1])

		elif(sys.argv[i] == '-mut'):
			mut = int(sys.argv[i+1])

		elif(sys.argv[i] == '-crs'):
			crs = int(sys.argv[i+1])	

		elif(sys.argv[i] == '-balance'):
			balance = int(sys.argv[i+1])

		elif(sys.argv[i] == '-train_type'):
			train_type = int(sys.argv[i+1])		

		elif(sys.argv[i] == '-kfold'):
			kfold = 1											

	CXPB = .8
	MUTPB = .2
	train_percent = 0.7
	
	filename = file_id + "GP_EEG_" + clf + str(param) + "_"	
	
	if len(opt_vars) == 0:
		opt_vars = ['auc']
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

	param = [param, param2]

	if kfold:
		for i, dt_op in zip([5,6,7,8], [25,26,27,28]):
			main(	NEXEC = i,
					classifier = clf,
					clf_param = param,
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
					wts_vars = wts_vars,
					ini = ini,
					sel = sel,
					mut = mut,
					crs = crs,
					balance = balance,
					train_type = train_type)		
	else:
		for i in execs:
			main(	NEXEC = i,
					classifier = clf,
					clf_param = param,
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
					wts_vars = wts_vars,
					ini = ini,
					sel = sel,
					mut = mut,
					crs = crs,
					balance = balance,
					train_type = train_type)
