
# coding: utf-8

# In[1]:


import os, shutil
import sys

if os.path.isdir("__pycache__"):
    shutil.rmtree("__pycache__")

from modules.balance_dataset import input_output
from modules.evaluation import fitness
from modules.utils import results_folder
from modules import visualize

from modules import operator_set
import operator

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import tools

from modules.statistics import runtime_metrics
import time
import numpy as np
import math


# In[2]:


DATASET_FOLDER = 'modules/data'
FILE_LIST = ['wav75_ex1_.csv','wav75_ex2_.csv','wav75_ex3_.csv','wav75_ex4_.csv','wav75_ex5_.csv','wav75_ex6_.csv','wav75_ex7_.csv','wav75_ex8_.csv']


# Importing data

# In[3]:


data = input_output.read_dataset_list(DATASET_FOLDER, FILE_LIST)
X_train, y_train, X_test, y_test = input_output.balance_dataset(data, data_columns = list(range(0,75)), label_column = 75, test_size = .2)

X_train = np.array(X_train)
X_test  = np.array(X_test)

# One hot vector
y_train = np.array([[x, int(not x)] for x in y_train])
y_test  = np.array([[x, int(not x)] for x in y_test])


# Select the evaluation metric

# In[4]:


opt_vars = ['auc']
eval_func = fitness.eval_function(opt_vars)
# Result: eval_func -> auc = lambda(y_true, y_pred)

# eval_func = fitness.eval_function(['auc', 'acc'])
## Result: eval_func -> auc, acc = lambda(y_true, y_pred)


# In[5]:


# Number of inputs
n_att = 75


# Select the classifier

# In[6]:


'''
    options:
        nb: Naive Bayes; params: None
        dt: Decision Tree; params: None
        mlp: Multilayer Perceptron; params: [[first hidden layer size],[activation function]]
        knn: K Nearest Neighbors; params: [[K],[-1]]
        svm: Support Vector Machine; params: [[-1],[kernel]]
        kmeans: K-means Clustering (Using each cluster as a class of the problem); params: [[number of clusters], [-1]]
'''
classifier = 'nb'
clf_param = [[-1],[-1]]


# GP operator set

# In[7]:


pset = gp.PrimitiveSet("MAIN", n_att)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator_set.plog, 1)
pset.addPrimitive(operator_set.psqrt, 1)
pset.addPrimitive(operator_set.pdiv, 2)
pset.addPrimitive(operator_set.F, 1)
# pset.addPrimitive(math.sin, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(operator.neg, 1)


# GP parameters

# In[8]:


'''
    Multiparameter optimization
    opt_vars -> evaluation metrics
    wts_vars -> tuple with each metric weight
    
    Example: Optimizing False positives and True positives
    opt_vars = [FP, TP]
    wts_vars = tuple([-1, 2])
    
    It means that we want to reduce False positives and maximize True positives. And it is more 
    important to have True positives, because we have set its weight higher.
'''
wts_vars = tuple([1])

creator.create("FitnessMulti", base.Fitness, weights = wts_vars)
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
# Individual and population
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Evaluation method
toolbox.register("evaluate", fitness.eval_tree, clf = classifier, param = clf_param, X_train = X_train, y_train = y_train, X_test = X_test, y_true = y_test, pset = pset, opt_vars = opt_vars, eval_func = eval_func)

# Initialization
# toolbox.register("expr_init", gp.genFull, min_=4, max_=7)
# toolbox.register("expr_init", gp.genGrow, min_=4, max_=7)
toolbox.register("expr_init", gp.genHalfAndHalf, min_=4, max_=7)

# Selection
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selRoulette)
# toolbox.register("select", tools.selRandom)

# Crossover
toolbox.register("mate", gp.cxOnePoint)
# toolbox.register("mate", gp.cxTwoPoint)
# toolbox.register("mate", gp.cxcxOnePointLeafBiased, termpb =.1)

# Mutation
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_init, pset=pset)
# toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
# toolbox.register("mutate", gp.mutInsert, pset=pset)
# toolbox.register("mutate", gp.mutShrink)toolbox.register("mutate", gp.mutEphemeral, mode = 'all')

# Bloat Control
TAM_MAX = 10
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = TAM_MAX))


# Statistics

# In[9]:


# random.seed(1)
start = time.time()
mstats = runtime_metrics.init_stats()


# GP start: Population 0

# In[10]:


# Population size
NPOP = 100

pop = toolbox.population(NPOP)

fitnesses = list(map(toolbox.evaluate, pop))
    
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
    if (math.isnan(fit[0])):
        ind.fitness.values = 0,
    else:
        ind.fitness.values = fit


# More statistics

# In[11]:


log = tools.Logbook()
hof = tools.selBest(pop, 1)


# GP start

# In[12]:


# Execution number (identifier)
NEXEC = 1

# Number of generations
NGEN = 50

# Crossover probability
CXPB = .85

# Mutation probability
MUTPB = .15

toolbar_width = 50
if NGEN < 50:
    toolbar_width = NGEN

# Verbosity level
verb = 1
if verb >= 1:
    print(">> (Exec " + str(NEXEC) + ") GP + " + classifier + " - Feature Selection and Construction")
    print(">> NGEN = " + str(NGEN) + " | NPOP = " + str(NPOP) + " | MAX_DEPTH = " + str(TAM_MAX) + " | PARAM = " + str(clf_param))
    print(">> Optimizing: " + str(opt_vars))
    print(">> Weights:    " + str(wts_vars))
    if verb == 1:
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1))

# Results folder
path = 'result_GP_' + str(NEXEC) + '/'
results_folder.verify_create_dir(path)

filename = 'GP_'+ classifier + '_' + 'NEXEC'
balance = 1

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

prf, acc, cfm, AUC = fitness.performance(hof[0], classifier, clf_param, X_train, y_train, X_test, y_test, pset)

info_file_name = os.path.join(path,"infoGP.csv")
infoGP = open(info_file_name, 'a')
if os.stat(info_file_name).st_size == 0:
    infoGP.write("balance,DEEP MAX,classifier,P1,P2,#Exec,PPV_S,PPV_NS,TPR_S,TPR_NS,F1_S,F1_NS,SUP_S,SUP_NS,TN,FP,FN,TP,Acc,AUC,Deep,Training Time\n")

infoGP.write(str(balance) + ',' + str(TAM_MAX) + ',' + classifier + ',' + str(clf_param[0]) + ',' + str(clf_param[1]) + ',' +  str(NEXEC) + ',' + str(prf[0][0]) + ',' 
        + str(prf[0][1]) + ',' + str(prf[1][0]) + ',' + str(prf[1][1]) + ',' + str(prf[2][0]) + ',' 
        + str(prf[2][1]) + ',' + str(prf[3][0]) + ',' + str(prf[3][1]) + ',' 
        + str(cfm[0]) + ',' + str(cfm[1]) + ',' + str(cfm[2]) + ',' + str(cfm[3]) + ',' 
        + str(acc) + ',' + str(AUC) + ',' + str(hof[0].height) + ',' + str(total_time) + '\n')

infoGP.close()

tree = gp.PrimitiveTree(hof[0])
expFILE = open(path + "best_expr/EXPR_" + filename + "_" +  str(NEXEC) + ".txt", 'w')
expFILE.write(str(tree))
expFILE.close()


# Visualize the best feature set created

# In[16]:


list(set(visualize.get_equations_simplified(hof[0])))


# Plot the GP tree

# In[17]:


visualize.plot_tree(hof[0], path = path + "best_expr/EXPR_" + filename + "_" +  str(NEXEC) + ".pdf")

