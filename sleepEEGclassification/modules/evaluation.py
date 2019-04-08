from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.cluster import KMeans

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from scipy.stats import wasserstein_distance

import pandas as pd
import numpy as np
from deap import gp

class fitness:
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

    def create_reduced_dataset(individual, pset, X):
        exp = gp.PrimitiveTree(individual)
        string = str(exp)
        ind = [i for i in range(len(string)) if string.startswith('F', i)]
        if len(ind) == 0:
            ind = [0]
        features = []
        hist = []
        temp = []
        for i in ind:
            subtree = fitness.get_subtree(i,string)
            if str(subtree) not in hist:
                hist.append(str(subtree))
                newtree = exp.from_string(subtree, pset)
                temp.append(str(newtree))
                features.append(gp.compile(newtree, pset))
        if len(features) == 0:
            features.append(gp.compile(individual, pset))
        X_new = []
        i = 0
        for x in X:
            X_new.append([])
            for feature, t in zip(features,temp):
                X_new[i].append(feature(*x))
            i += 1
        return X_new

    
    def feature_construction(individual, clf, X_train, y_train, X_test, pset, param = [[],[]], return_proba = False, external = False):
        X_train_new = fitness.create_reduced_dataset(individual, pset, X_train)

        if external != False:
            classifier = external
        elif clf == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=param[0])
        elif clf == 'mlp':
            classifier = MLP(hidden_layer_sizes=(param[0], ), activation=param[1], max_iter = 200)
        elif clf == 'svm':
            classifier = SVC(kernel = param[1], probability=True)
        elif clf == 'dt':
            classifier = DT()
        elif clf == 'nb':
            classifier = GaussianNB()
        elif clf == 'kmeans':
            classifier = KMeans(n_clusters=param[0])

        y_train = np.array([j[0] for j in y_train])

        X_train_new = np.array(X_train_new).astype(np.float)
        try:
            classifier.fit(X_train_new, y_train)
        except:
            return -1
            
        X_test_new = fitness.create_reduced_dataset(individual, pset, X_test)

        X_test_new = np.array(X_test_new).astype(np.float)
        y_pred = classifier.predict(X_test_new)
        y_pred = np.array([[j, int(not(j))] for j in y_pred])

        if return_proba:
                predict_proba = classifier.predict_proba(X_test_new)
                return y_pred, predict_proba
        else:
            return y_pred

    def eval_function(opt_vars):
        # prf = precision_recall_fscore_support(y_true, y_pred)
        # acc = accuracy_score(y_true, y_pred)
        # cfm = confusion_matrix(y_true, y_pred)
        
        if 'wd' in opt_vars:
            return lambda X, y: [[wasserstein_distance(X[y[0] == 1][i].values, X[y[0] == 0][i].values) for i in range(len(X.columns))]]
        
        func = []
        metr1 = []
        metr2 = []
        if 'acc' in opt_vars:
            func.append(0)
        if 'auc' in opt_vars:
            func.append(3)
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
        if 3 in func:
            w = lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
            funcs.append(w)
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

    
    
    
    def eval_tree(individual, clf, X_train, y_train, X_test, y_true, pset, opt_vars, eval_func, param = [[],[]], external = False):
        y_pred = fitness.feature_construction(individual, clf, X_train, y_train, X_test, pset, param, external)

        if 'wd' in opt_vars:
            X = fitness.create_reduced_dataset(individual, pset, X_test)
            if len(X[0]) < 2:
                return 0,
            res = eval_func(pd.DataFrame(X), pd.DataFrame(y_true))
            print(res)
            print()
            ret = tuple([np.mean(res)])
            return ret
        
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

    def performance(individual, clf, X_train, y_train, X_test, y_true, pset, param = [[],[]], external = False):
        ret = fitness.feature_construction(individual, clf, X_train, y_train, X_test, pset, param = param, return_proba = True, external = external)
        if type(ret) == type(-1):
            return [[0,0]]*4, 0, [0]*4, 0
        
        y_pred, predict_proba = ret[0], ret[1]

        # fpr = dict()
        # tpr = dict()
        # AUC = dict()
        # for c in range(2):
        #     fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_pred[:, c])
        #     AUC[c] = auc(fpr[c], tpr[c])
        # AUC = AUC[0]

        
        try:
            AUC = roc_auc_score(y_true, y_pred)
        except:
            AUC = 0
    
        if AUC < 0.5:
            AUC = 1 - AUC            
                
        #	y_pred = [[i[1], i[0]] for i in y_pred]
        #	fpr = dict()
        #	tpr = dict()
        #	AUC = dict()
        #	for c in range(2):
        #	    fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_pred[:, c])
        #	    AUC[c] = auc(fpr[c], tpr[c])			
        prf = precision_recall_fscore_support(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        y_true = [i[0] for i in y_true]
        y_pred = [i[0] for i in y_pred]
        
        cfm = confusion_matrix(y_true, y_pred).ravel()
        return prf, acc, cfm, AUC