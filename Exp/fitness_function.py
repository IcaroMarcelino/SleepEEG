from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
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

def knn_feature_selection(individual, K, X_train, y_train, X_test, y_test, toolbox, pset):
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

	pred = knn.predict(X_test_new)
	return pred

def eval_tree(individual, K, X_train, y_train, X_test, y_test, toolbox, pset, opt_vars):
	pred = knn_feature_selection(individual, K, X_train, y_train, X_test, y_test, toolbox, pset)

	if type(pred) == type(-1):
		ret = tuple([0]*len(opt_vars))
		return ret
	
	prf = precision_recall_fscore_support(y_test, pred)
	acc = accuracy_score(y_test, pred)
	
	ret = []
	for var in opt_vars:
		if var == 'acc':
			ret.append(acc)
		elif var == 'f1_S':
			ret.append(prf[2][0])
		elif var == 'f1_NS':
			ret.append(prf[2][1])
		elif var == 'prec_S':
			ret.append(prf[0][0])
		elif var == 'rec_S':
			ret.append(prf[0][1])
		elif var == 'prec_NS':
			ret.append(prf[1][0])
		elif var == 'rec_NS':
			ret.append(prf[1][1])

	return tuple(ret)

def performance(individual, K, X_train, y_train, X_test, y_test, toolbox, pset):
	pred = knn_feature_selection(individual, K, X_train, y_train, X_test, y_test, toolbox, pset)

	if type(pred) == type(-1):
		return [0]*8, 0
	
	prf = precision_recall_fscore_support(y_test, pred)
	acc = accuracy_score(y_test, pred)

	return prf, acc