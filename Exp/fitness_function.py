from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from deap import gp

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

def knn_feature_selection(individual, K, X_train, y_train, X_test, toolbox, pset):
	exp = gp.PrimitiveTree(individual)
	string = str(exp)
	ind = [i for i in range(len(string)) if string.startswith('F', i)]
	features = []
	hist = []
	for i in ind:
		subtree = get_subtree(i,string)
		if subtree not in hist:
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
	knn = KNeighborsClassifier(n_neighbors=K)
	try:
		knn.fit(X_train_new, y_train)
	except:
		return -1
	X_test_new = []
	i = 0
	for x in X_test:
		X_test_new.append([])
		for feature in features:
			X_test_new[i].append(feature(*x))
		i += 1

	y_pred = knn.predict(X_test_new)
	return y_pred

def eval_function(opt_vars):
	# prf = precision_recall_fscore_support(y_true, y_pred)
	# acc = accuracy_score(y_true, y_pred)
	# cfm = confusion_matrix(y_true, y_pred)
	func = []
	metr1 = []
	metr2 = []
	if 'acc' in opt_vars:
		func.append(0)
	if 'prec_S' in opt_vars:
		func.append(1)
		metr1.append((0,0))
	if 'prec_NS' in opt_vars:
		func.append(1)
		metr1.append((0,1))
	if 'rec_S' in opt_vars:
		func.append(1)
		metr1.append((1,0))
	if 'rec_NS' in opt_vars:
		func.append(1)
		metr1.append((1,1))
	if 'f1_S' in opt_vars:
		func.append(1)
		metr1.append((2,0))
	if 'f1_NS' in opt_vars:
		func.append(1)
		metr1.append((2,1))
	if 'TN' in opt_vars:
		func.append(2)
		metr2.append(0)
	if 'FP' in opt_vars:
		func.append(2)
		metr2.append(1)
	if 'FN' in opt_vars:
		func.append(2)
		metr2.append(2)
	if 'TP' in opt_vars:
		func.append(2)
		metr2.append(3)
	funcs = []
	if 0 in func:
		x = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
		funcs.append(x)
	if 1 in func:
		y = lambda y_true, y_pred: [precision_recall_fscore_support(y_true, y_pred)[i][j] for i,j in metr1]
		funcs.append(y)
	if 2 in func:
		z = lambda y_true, y_pred: [confusion_matrix([j[0] for j in y_true], [k[0] for k in y_pred]).ravel()[i] for i in metr2]
		funcs.append(z)
	final_func = lambda y_true, y_pred: [f(y_true, y_pred) for f in funcs]
	return final_func

def eval_tree(individual, K, X_train, y_train, X_test, y_true, toolbox, pset, opt_vars, eval_func):
	y_pred = knn_feature_selection(individual, K, X_train, y_train, X_test, toolbox, pset)

	if type(y_pred) == type(-1):
		ret = tuple([0]*len(opt_vars))
		return ret

	ret = []
	res = eval_func(y_true, y_pred)
	for item in res:
		if type(item) == list:
			for i in item:
				ret.append(i)
		else:
			ret.append(item)

	return tuple(ret)

def performance(individual, K, X_train, y_train, X_test, y_true, toolbox, pset):
	y_pred = knn_feature_selection(individual, K, X_train, y_train, X_test, toolbox, pset)

	if type(y_pred) == type(-1):
		return [0]*8, 0
	
	prf = precision_recall_fscore_support(y_true, y_pred)
	acc = accuracy_score(y_true, y_pred)

	y_true = [i[0] for i in y_true]
	y_pred = [i[0] for i in y_pred]
	
	cfm = confusion_matrix(y_true, y_pred).ravel()

	return prf, acc, cfm