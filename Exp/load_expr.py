import pylab as pl
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from fitness_function import*
from input_output import*
import pygraphviz as pgv
from sympy import simplify, expand
from deap import gp
import operator
from sklearn import metrics
from sklearn.manifold import TSNE

from operator_set import*
import math
import numpy as np
import pandas as pd

def psqrt(x):
	return math.sqrt(abs(x))

def div(left, right):
	try:
		return eval(str(left))/eval(str(right))
	except ZeroDivisionError:
		return 1

def plog(x):
	try:
		return math.log(abs(x))
	except:
		return 1

def F(x):
	return x


def convertFunct(string):
	def add(a,b):
		return('({} + {})'.format(a,b))

	def neg(a):
		return('(-{})'.format(a))

	def plog(a):
		return('plog({})'.format(a))

	def psqrt(a):
		return('psqrt({})'.format(a))

	def sub(a,b):
		return('({} - {})'.format(a,b))

	def pdiv(a,b):
		return('({} * {}^(-1))'.format(a,b))

	#def div(a,b):
	#	return('({} / {})'.format(a,b))

	#def div(a,b):
	#	return('pdiv({}, {})'.format(a,b))

	def mul(a,b):
		return('({} * {})'.format(a,b))

	def F(a):
		return('{}'.format(a))

	ARG0 = 'ARG0'
	ARG1 = 'ARG1'
	ARG2 = 'ARG2'
	ARG3 = 'ARG3'
	ARG4 = 'ARG4'
	ARG5 = 'ARG5'
	ARG6 = 'ARG6'
	ARG7 = 'ARG7'
	ARG8 = 'ARG8'
	ARG9 = 'ARG9'
	ARG10 = 'ARG10'
	ARG11 = 'ARG11'
	ARG12 = 'ARG12'
	ARG13 = 'ARG13'
	ARG14 = 'ARG14'
	ARG15 = 'ARG15'
	ARG16 = 'ARG16'
	ARG17 = 'ARG17'
	ARG18 = 'ARG18'
	ARG19 = 'ARG19'
	ARG20 = 'ARG20'
	ARG21 = 'ARG21'
	ARG22 = 'ARG22'
	ARG23 = 'ARG23'
	ARG24 = 'ARG24'
	ARG25 = 'ARG25'
	ARG26 = 'ARG26'
	ARG27 = 'ARG27'
	ARG28 = 'ARG28'
	ARG29 = 'ARG29'
	ARG30 = 'ARG30'
	ARG31 = 'ARG31'
	ARG32 = 'ARG32'
	ARG33 = 'ARG33'
	ARG34 = 'ARG34'
	ARG35 = 'ARG35'
	ARG36 = 'ARG36'
	ARG37 = 'ARG37'
	ARG38 = 'ARG38'
	ARG39 = 'ARG39'
	ARG40 = 'ARG40'
	ARG41 = 'ARG41'
	ARG42 = 'ARG42'
	ARG43 = 'ARG43'
	ARG44 = 'ARG44'
	ARG45 = 'ARG45'
	ARG46 = 'ARG46'
	ARG47 = 'ARG47'
	ARG48 = 'ARG48'
	ARG49 = 'ARG49'
	ARG50 = 'ARG50'
	ARG51 = 'ARG51'
	ARG52 = 'ARG52'
	ARG53 = 'ARG53'
	ARG54 = 'ARG54'
	ARG55 = 'ARG55'
	ARG56 = 'ARG56'
	ARG57 = 'ARG57'
	ARG58 = 'ARG58'
	ARG59 = 'ARG59'
	ARG60 = 'ARG60'
	ARG61 = 'ARG61'
	ARG62 = 'ARG62'
	ARG63 = 'ARG63'
	ARG64 = 'ARG64'
	ARG65 = 'ARG65'
	ARG66 = 'ARG66'
	ARG67 = 'ARG67'
	ARG68 = 'ARG68'
	ARG69 = 'ARG69'
	ARG70 = 'ARG70'
	ARG71 = 'ARG71'
	ARG72 = 'ARG72'
	ARG73 = 'ARG73'
	ARG74 = 'ARG74'
	ARG75 = 'ARG75'
	ARG76 =	'ARG76'
	ARG77 =	'ARG77'
	ARG78 =	'ARG78'
	ARG79 =	'ARG79'
	ARG80 =	'ARG80'
	ARG81 =	'ARG81'
	ARG82 =	'ARG82'
	ARG83 =	'ARG83'
	ARG84 =	'ARG84'
	ARG85 =	'ARG85'
	ARG86 =	'ARG86'
	ARG87 =	'ARG87'
	ARG88 =	'ARG88'
	ARG89 =	'ARG89'
	ARG90 =	'ARG90'
	ARG91 =	'ARG91'
	ARG92 =	'ARG92'
	ARG93 =	'ARG93'
	ARG94 =	'ARG94'
	ARG95 =	'ARG95'
	ARG96 =	'ARG96'
	ARG97 =	'ARG97'
	ARG98 =	'ARG98'
	ARG99 =	'ARG99'
	ARG100	=	'ARG100'
	ARG101	=	'ARG101'
	ARG102	=	'ARG102'
	ARG103	=	'ARG103'
	ARG104	=	'ARG104'
	ARG105	=	'ARG105'
	ARG106	=	'ARG106'
	ARG107	=	'ARG107'
	ARG108	=	'ARG108'
	ARG109	=	'ARG109'
	ARG110	=	'ARG110'
	ARG111	=	'ARG111'
	ARG112	=	'ARG112'
	ARG113	=	'ARG113'
	ARG114	=	'ARG114'
	ARG115	=	'ARG115'
	ARG116	=	'ARG116'
	ARG117	=	'ARG117'
	ARG118	=	'ARG118'
	ARG119	=	'ARG119'
	ARG120	=	'ARG120'
	ARG121	=	'ARG121'
	ARG122	=	'ARG122'
	ARG123	=	'ARG123'
	ARG124	=	'ARG124'
	ARG125	=	'ARG125'
	ARG126	=	'ARG126'
	ARG127	=	'ARG127'
	ARG128	=	'ARG128'
	ARG129	=	'ARG129'
	ARG130	=	'ARG130'
	ARG131	=	'ARG131'
	ARG132	=	'ARG132'
	ARG133	=	'ARG133'
	ARG134	=	'ARG134'
	ARG135	=	'ARG135'
	ARG136	=	'ARG136'
	ARG137	=	'ARG137'
	ARG138	=	'ARG138'
	ARG139	=	'ARG139'
	ARG140	=	'ARG140'
	ARG141	=	'ARG141'
	ARG142	=	'ARG142'
	ARG143	=	'ARG143'
	ARG144	=	'ARG144'
	ARG145	=	'ARG145'
	ARG146	=	'ARG146'
	ARG147	=	'ARG147'
	ARG148	=	'ARG148'
	ARG149	=	'ARG149'
	ARG150	=	'ARG150'
	ARG151	=	'ARG151'
	ARG152	=	'ARG152'
	ARG153	=	'ARG153'
	ARG154	=	'ARG154'
	ARG155	=	'ARG155'
	ARG156	=	'ARG156'
	ARG157	=	'ARG157'
	ARG158	=	'ARG158'
	ARG159	=	'ARG159'
	ARG160	=	'ARG160'
	ARG161	=	'ARG161'
	ARG162	=	'ARG162'
	ARG163	=	'ARG163'
	ARG164	=	'ARG164'
	ARG165	=	'ARG165'

	return eval(string)

def convertFunct_pretty(string):
	ARG0 = 'x0'
	ARG1 = 'x1'
	ARG2 = 'x2'
	ARG3 = 'x3'
	ARG4 = 'x4'
	ARG5 = 'x5'
	ARG6 = 'x6'
	ARG7 = 'x7'
	ARG8 = 'x8'
	ARG9 = 'x9'
	ARG10 = 'x10'
	ARG11 = 'x11'
	ARG12 = 'x12'
	ARG13 = 'x13'
	ARG14 = 'x14'
	ARG15 = 'x15'
	ARG16 = 'x16'
	ARG17 = 'x17'
	ARG18 = 'x18'
	ARG19 = 'x19'
	ARG20 = 'x20'
	ARG21 = 'x21'
	ARG22 = 'x22'
	ARG23 = 'x23'
	ARG24 = 'x24'
	ARG25 = 'x25'
	ARG26 = 'x26'
	ARG27 = 'x27'
	ARG28 = 'x28'
	ARG29 = 'x29'
	ARG30 = 'x30'
	ARG31 = 'x31'
	ARG32 = 'x32'
	ARG33 = 'x33'
	ARG34 = 'x34'
	ARG35 = 'x35'
	ARG36 = 'x36'
	ARG37 = 'x37'
	ARG38 = 'x38'
	ARG39 = 'x39'
	ARG40 = 'x40'
	ARG41 = 'x41'
	ARG42 = 'x42'
	ARG43 = 'x43'
	ARG44 = 'x44'
	ARG45 = 'x45'
	ARG46 = 'x46'
	ARG47 = 'x47'
	ARG48 = 'x48'
	ARG49 = 'x49'
	ARG50 = 'x50'
	ARG51 = 'x51'
	ARG52 = 'x52'
	ARG53 = 'x53'
	ARG54 = 'x54'
	ARG55 = 'x55'
	ARG56 = 'x56'
	ARG57 = 'x57'
	ARG58 = 'x58'
	ARG59 = 'x59'
	ARG60 = 'x60'
	ARG61 = 'x61'
	ARG62 = 'x62'
	ARG63 = 'x63'
	ARG64 = 'x64'
	ARG65 = 'x65'
	ARG66 = 'x66'
	ARG67 = 'x67'
	ARG68 = 'x68'
	ARG69 = 'x69'
	ARG70 = 'x70'
	ARG71 = 'x71'
	ARG72 = 'x72'
	ARG73 = 'x73'
	ARG74 = 'x74'
	ARG75 = 'x75'
	return eval(string)

def get_equations_simplified(individual):
	string = str(individual)
	ind = [i for i in range(len(string)) if string.startswith('F', i)]
	if len(ind) == 0:
		ind = [0]
	features = []
	for i in ind:
		subtree = get_subtree(i,string)
		features.append(subtree)
	if len(features) == 0:
		features.append(string)
	eqs = []
	for eq in features:
		#print(eq, simplify(convertFunct(eq)))
		temp = expand(simplify(convertFunct(eq)))
		#eqs.append(convertFunct(eqs))
		eqs.append(temp)
	return list(set(eqs))

def individual_pdf(file_path, individual, pset):
	nodes, edges, labels = gp.graph(individual)
	g = pgv.AGraph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	g.layout(prog='dot')
	for i in nodes:
		n = g.get_node(i)
		n.attr['label'] = labels[i]
	g.draw(file_path, format='png',prog='dot')
	return

def init_pset(n_att):
	pset = gp.PrimitiveSet("MAIN", n_att)
	pset.addPrimitive(operator.add, 2)
	pset.addPrimitive(operator.sub, 2)
	pset.addPrimitive(operator.mul, 2)
	pset.addPrimitive(plog, 1)
	pset.addPrimitive(psqrt, 1)
	pset.addPrimitive(F, 1)
	pset.addPrimitive(pdiv, 2)
	pset.addPrimitive(div, 2)
	return pset


def load_model(n_att, file_name):
	pset = init_pset(n_att)
	file = open(file_name, 'r')
	string = file.read()
	#print(string)
	#string = string.replace('div', 'pdiv')
	#string = string.replace('sqrt', 'psqrt')
	#string = string.replace('log', 'plog')
	expr = gp.genFull(pset, min_=1, max_=3)
	tree = gp.PrimitiveTree(expr)
	ind  = tree.from_string(string, pset)
	file.close()
	return ind

def get_prediction_from_expr(n_att, classifier, param, file_name, files_train, file_test):
	pset = init_pset(n_att)
	#ind = txt_to_individual(file_name, pset)
	file = open(file_name, 'r')
	string = file.read()
	#print(string)
	#string = string.replace('div', 'pdiv')
	#string = string.replace('sqrt', 'psqrt')
	#string = string.replace('log', 'plog')
	expr = gp.genFull(pset, min_=1, max_=3)
	tree = gp.PrimitiveTree(expr)
	ind  = tree.from_string(string, pset)
	file.close()
	#t, tt, xt, yt, ttt = import_all_data(files_train, 1, 0, 0, 1)
	#, tt, x1, y1, ttt = import_data(file_test, 0, 0)
	xt, yt, x1, y1, n_att = import_all_data(files_train,1, .2, 1, 0)
	#print(str(ind))
	pred = feature_construction(ind, classifier, param, xt, yt, x1, pset)
	pred = [i[0] for i in pred]
	y1   = [i[0] for i in y1]
	return pred, y1

def data_set_transform(individual, pset, X):
	exp = gp.PrimitiveTree(individual)
	string = str(exp)
	ind = [i for i in range(len(string)) if string.startswith('F', i)]
	if len(ind) == 0:
		ind = [0]
	features = []
	hist = []
	for i in ind:
		subtree = get_subtree(i,string)
		if subtree not in hist:
			newtree = exp.from_string(subtree, pset)
			features.append(gp.compile(newtree, pset))
	if len(features) == 0:
		features.append(gp.compile(individual, pset))
	X_new = []
	features = list(set(features))
	i = 0
	for x in X:
		X_new.append([])
		for feature in features:
			#print(x)
			#print(str(features))
			X_new[i].append(feature(*x))
		i += 1
	return list(np.array(X_new).astype(np.float))

def elbow_method_kmeans(X, Y, Y_pred, max_Nc, clf, folder, Xall = [], Yall = []):
	Nc = range(1, max_Nc)
	kmeans = [KMeans(n_clusters=i) for i in Nc]
	score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]

	pl.figure(1)
	pl.plot(Nc,score)
	pl.xlabel('Número de Clusters')
	pl.ylabel('Score')
	pl.title('Método Elbow (dataset ' + folder + ')\nAtributos GP + ' + clf.upper())

	gs = gridspec.GridSpec(1, 3)
	pl.figure(2)
	ax = pl.subplot(gs[0, 0])
	if(len(Xall) > 0):
		print("TSNE 1")
		tsne_all = TSNE(n_components=2)
		X_2d_all = tsne_all.fit_transform(Xall)
		target_ids = [0, 1]
		#pl.figure(figsize=(6, 5))
		colors = 'r', 'b'

		for i, c, label in zip(target_ids, colors, [0, 1]):
			pl.scatter(X_2d_all[Yall == i, 0], X_2d_all[Yall == i, 1], c=c, label=label)
		
		pl.title('(dataset original)\nClusters GP + ' + clf.upper() + '\n Predição')
		pl.legend()
		pl.xlim(np.array(X_2d_all).min(), np.array(X_2d_all).max())
		pl.ylim(np.array(X_2d_all).min(), np.array(X_2d_all).max())

	y = kmeans[1].fit(X).labels_
	print("TSNE 2")
	tsne = TSNE(n_components=2)
	X_2d = tsne.fit_transform(X)

	ax = pl.subplot(gs[0, 1])
	target_ids = [0, 1]
	#pl.figure(figsize=(6, 5))
	colors = 'r', 'b'
	for i, c, label in zip(target_ids, colors, [0, 1]):
		pl.scatter(X_2d[Y_pred == i, 0], X_2d[Y_pred == i, 1], c=c, label=label)
		
	pl.title('(dataset ' + folder + ')\nClusters GP + ' + clf.upper() + '\n Predição')
	pl.legend()
	pl.xlim(np.array(X_2d).min(), np.array(X_2d).max())
	pl.ylim(np.array(X_2d).min(), np.array(X_2d).max())

	if sum(Y) > sum(np.logical_not(Y)):
		Y = np.array(np.logical_not(Y), dtype = 'int32')

	ax = pl.subplot(gs[0, 2])
	for i, c, label in zip(target_ids, colors, [0, 1]):
		pl.scatter(X_2d[Y == i, 0], X_2d[Y == i, 1], c=c, label=label)
		
	pl.title('(dataset ' + folder + ')\nClusters GP + ' + clf.upper() + '\n Labels originais')
	pl.legend()
	pl.xlim(np.array(X_2d).min(), np.array(X_2d).max())
	pl.ylim(np.array(X_2d).min(), np.array(X_2d).max())
	pl.show()

	return y

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


def media(M, X, K):
	return M*(K-1)/K + X/K

def desv_padrao(S, M, X, K):
	return ((S**2 + ((M-X)**2)/(K+1))*K/(K+1))**.5

d1 = []
d2 = []
d3 = []


for folder, dataset, n_att in zip(['T10_2b', 'T10_01b', 'TFFb'], [files_wav75, files_wav75_00, files_wav75_FF], [75, 75, 75]):
	d  = [[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
	 	[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
		[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
		[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],
		[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]]
	
	for file in os.listdir('BEST_FEATURES/' + folder):
		print(file)
		if 'dt' in file:
			index1 = 0
			clf = 'dt'
			p1  = -1
			p2  = ''
		elif 'svm' in file:
			index1 = 1
			clf = 'svm'
			p1  = -1
			p2  = 'rbf'
		elif 'nb' in file:			
			index1 = 2
			clf = 'nb'
			p1  = -1
			p2  = ''
		elif 'knn' in file:
			index1 = 3
			clf = 'knn'
			p1  = 5
			p2  = ''
		elif 'mlp' in file:
			index1 = 4			
			clf = 'mlp'
			p1  = 15
			p2  = 'relu'

		for clf_, param, index2 in zip(['dt', 'nb', 'knn', 'svm', 'mlp'], [[-1, ''], [-1, ''], [5, ''], [-1, 'rbf'], [15, 'relu']], [0,1,2,3,4]):
			pset = init_pset(n_att)
			ind = load_model(n_att, 'BEST_FEATURES/' + folder + '/' + file)
				
			X_train, y_train, X_test, y_true, _ = import_all_data(dataset, 1, .7, 1, 0)
			#X_train = data_set_transform(ind, pset, X_train)
			#X_test  = data_set_transform(ind, pset, X_test)

			prf, acc, cfm, AUC = performance(ind, clf_, param, X_train, y_train, X_test, y_true, pset)

			print("Dataset:", folder)
			print("Classificador original:", clf)
			print("Classificador teste   :", clf_)
			print("F1 :", str(prf[2][0]))
			print("AUC:", AUC)
			print("ACC:", acc)

			M = d[index1][index2][0]
			S = d[index1][index2][1]
			K = d[index1][index2][2]

			d[index1][index2][0] = media(M, AUC, K)
			d[index1][index2][1] = desv_padrao(S, M, AUC, K)
			d[index1][index2][2] = K+1
			
			print(d[index1][index2])
			print()

	
	print(np.array(d))
	input()
 print(d)

folder = 'TKM'
dataset = files_wav75
clf = 'kmeans'
n_att = 75
p1 = 2
p2 = ''
file_name = os.listdir('BEST_FEATURES/' + folder)[0]



#y_pred,  y_true = get_prediction_from_expr(75, clf, [p1, p2], 'BEST_FEATURES/' + folder + '/' + file, dataset,'')
#for folder, dataset, n_att in zip(['T10_2b', 'T10_01b', 'TFFb', 'TF11b', 'TStpb'], [files_wav75, files_wav75_00, files_wav75_FF, files_wav75_F11, files_wav75_FN], [75, 75, 75, 165, 75]):
#for folder, dataset, n_att in zip(['T10_2b', 'T10_01b'], [files_wav75, files_wav75_00], [75, 75]):
#	for file in os.listdir('BEST_FEATURES/' + folder):
#		if 'dt' in file:
#			clf = 'dt'
#			p1  = -1
#			p2  = ''
#		elif 'svm' in file:
#			clf = 'svm'
#			p1  = -1
#			p2  = 'rbf'
#		else:
#			break
#
#		#elif 'nb' in file:
#		#	clf = 'nb'
#		#	p1  = -1
#		#	p2  = ''
#		#elif 'knn' in file:
#		#	clf = 'knn'
#		#	p1  = 5
#		#	p2  = ''
#		
#		#elif 'mlp' in file:
#		#	clf = 'mlp'
#		#	p1  = 15
#		#	p2  = 'relu'
		
clf = 'dt'
p1  = -1
p2  = ''
n_att = 75
folder = 'T10_2b'
file_name = 'EXPR_T10_2bGP_EEG_svm5__25_20.txt'
dataset = files_wav75




#y_pred,  y_true = get_prediction_from_expr(n_att, clf, [p1, p2], 'BEST_FEATURES/' + folder + '/' + file, dataset,'')
#print(np.array(y_pred)&np.array(y_true))
#print(sum(y_pred), sum(y_true), sum(np.array(y_pred)&np.array(y_true)))
#print(y_pred)
#ind = load_model(n_att, 'BEST_FEATURES/' + folder + '/' + file)

pset = init_pset(n_att)
X1, Y1, X2, Y2, _ = import_all_data(dataset,0, 0.0, 0, 1)
#_, _, _, Y2, _ = import_all_data(dataset,0, 0, 0, 1, 1)

X1 = X2
Y1 = Y2
pset = init_pset(n_att)
file = open('BEST_FEATURES/' + folder + '/' +file_name, 'r')
string = file.read()
expr = gp.genFull(pset, min_=1, max_=3)
tree = gp.PrimitiveTree(expr)
ind  = tree.from_string(string, pset)
file.close()

X1t = data_set_transform(ind, pset, X1)
X2t = data_set_transform(ind, pset, X2)

y_pred = feature_construction(ind, clf, [2, ''], X1, Y2, X2, pset)
y_pred = [i[0] for i in y_pred]

k_pred = elbow_method_kmeans(X2t, np.array([y[0] for y in Y2], dtype= 'int'), np.array(y_pred, dtype= 'int'), 10, clf, folder, X1, np.array([y[0] for y in Y1]))




# print(len(X), len(X[0]))
# X = data_set_transform(ind, pset, X)
# print(len(X), len(X[0]))
# k_pred = elbow_method_kmeans(X, np.array([y[0] for y in Y2], dtype= 'int'), 20, clf, folder)

# k_pred = np.array(k_pred, dtype = 'int32')
# y_ = np.array([y[0] for y in Y2], dtype = 'int32')
# print(sum(y_), sum(k_pred))
# print()

# print("Spindles 1: ", sum(np.logical_not(np.logical_xor(y_,k_pred)))/len(y_), sum(k_pred), sum(y_), len(y_))
# print("Spindles 2: ", sum(np.logical_not(np.logical_xor(y_,np.logical_not(k_pred))))/len(y_))


# X = [np.concatenate((x, np.array([y1[0], y2[0]], dtype = 'int')),axis=0) for x,y1,y2 in zip(X,Y1,Y2)]
# print(len(X), len(X[0]))
# eqs = get_equations_simplified(ind)

# df = pd.DataFrame(X)
# df.to_csv('BEST_FEATURES_SIMPLIFIED/' + folder + '/' + file[:-4] + '_NEWDATASET.csv', index = False, header = False, index_label = False)

# simplified = open('BEST_FEATURES_SIMPLIFIED/' + folder + '/' + file[:-4] + 'SIMPLIFIED.txt', 'w')

# for eq in eqs:
# 	simplified.write(str(eq) + '\n')

# simplified.close() 

# print(folder, file)
# print("Resultado:\n%s\n" % (metrics.classification_report(y_true, y_pred)))
# #input()
# #print("Pred from ", file_test)
# #print(y_pred)
# #print(sum(y_pred), sum(y_true))
# #print((np.logical_not(np.array(y_pred)^np.array(y_true))*1))
# #result = (np.array(y_pred)&np.array(y_true))*1+np.array(y_pred)

# #resumo = []
# #for r in result:
# #	if r == 0:
# #		resumo.append(' ')
# #	if r == 1:
# #		resumo.append('s')
# #	if r == 2:
# #		resumo.append('S')
# #print('--')
# #print(''.join(resumo))
# #print('--')

# #resumo = []
# #for r, t in zip(np.array(y_pred), np.array(y_true)):
# #	if r == 1 and t == 1:
# #		resumo.append('S')
# #	if r == 0 and t == 0:
# #		resumo.append(' ')
# #	if r == 1 and t == 0:
# #		resumo.append('s')
# #	if r == 0 and t == 1:
# #		resumo.append('o')
# #print('--')
# #print(''.join(resumo))
# #print('--')