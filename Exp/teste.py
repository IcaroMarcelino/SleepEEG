import pygraphviz as pgv
from sympy import simplify, expand
from deap import gp
import operator

from fitness_function import get_subtree
from operator_set import*

def sqrt(x):
	return abs(x)**2


def div(left, right):
	try:
		return left / right
	except ZeroDivisionError:
		return 1

def log(x):
	try:
		return math.log(abs(x))
	except:
		return 1

def convertFunct(string):
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
	features = []
	for i in ind:
		subtree = get_subtree(i,string)
		features.append(subtree)
	if len(features) == 0:
		features.append(string)
	eqs = []
	for eq in features:
		temp = expand(simplify(convertFunct(eq)))
		#eqs.append(convertFunct(eqs))
		eqs.append(temp)
	return eqs

def txt_to_individual(file_path, pset):
	file = open(file_path, 'r')
	string = file.read()
	file.close()

	string.replace('div', 'pdiv')
	string.replace('sqrt', 'psqrt')
	string.replace('log', 'plog')
	expr = gp.genFull(pset, min_=1, max_=3)
	tree = gp.PrimitiveTree(expr)
	individual = tree.from_string(string, pset)
	return individual

def individual_pdf(file_path, individual, pset):
	nodes, edges, labels = gp.graph(individual)
	g = pgv.AGraph()
	g.add_nodes_from(nodes)
	g.add_edges_from(edges)
	g.layout(prog='dot')
	for i in nodes:
		n = g.get_node(i)
		n.attr['label'] = labels[i]
	g.draw(file_path + '.pdf')
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