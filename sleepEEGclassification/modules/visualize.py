import pygraphviz as pgv
from sympy import simplify, expand
from deap import gp
import operator

from .evaluation import fitness
from .operator_set import *

import matplotlib.pyplot as plt
import networkx as nx

def add(a,b):
    return('({} + {})'.format(a,b))

def neg(a):
    return('(-{})'.format(a))

def plog(a):
    return('log({})'.format(a))

def psqrt(a):
    return('sqrt({})'.format(a))

def sub(a,b):
    return('({} - {})'.format(a,b))

def pdiv(a,b):
    return('({} / {})'.format(a,b))

def div(a,b):
    return('({} / {})'.format(a,b))

def mul(a,b):
    return('({} * {})'.format(a,b))

def F(a):
    return('{}'.format(a))

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

def get_equations_simplified(individual):
    string = str(individual)
    ind = [i for i in range(len(string)) if string.startswith('F', i)]
    features = []
    for i in ind:
        subtree = fitness.get_subtree(i,string)
        features.append(subtree)
    if len(features) == 0:
        features.append(string)
    eqs = []
    for eq in features:
        temp = expand(simplify(convertFunct(eq)))
        eqs.append(temp)
    return eqs
    

def plot_tree(individual, save_file = True, path = 'tree.pdf'):
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos, node_color='lightgray',node_size=3000)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels, font_size=16)

    plt.rcParams["figure.figsize"] = [25,16]
    plt.margins(.1,.1)    
    plt.axis('off')
    
    if save_file:
        plt.savefig(path)
    
    plt.show()
    return