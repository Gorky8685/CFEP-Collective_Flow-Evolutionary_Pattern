import sys,os

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt 
import CFEP
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from numpy import random
def sum_weight(g_list):
    weight_list=[]
    for g in g_list:
        total_weight=0
        for edge in g.edges(data=True):
            total_weight+=edge[2]['weight']
        weight_list.append(total_weight)

    return weight_list

def com_dict2list(com_dict):
    com_list=[]
    for i in set(com_dict.values()):
        com_list.append([])
    for key,value in com_dict.items():
        com_list[value].append(key)
    return com_list

def ini_graph_weight(G_t1,G_t2,ini_weight):
    weights1=[ini_weight for _ in range(len(G_t1.edges()))]
    weights2=[ini_weight for _ in range(len(G_t1.edges()))]

    for edge,weight in zip(G_t1.edges(),weights1):
        G_t1[edge[0]][edge[1]]['weight']=weight

    for edge,weight in zip(G_t1.edges(),weights2):
        G_t2[edge[0]][edge[1]]['weight']=weight

    # Increase one block with uniform increasing rate 2
    B1_G_t2=G_t2.subgraph(G_t2.graph['partition'][0])
    B2_G_t2=G_t2.subgraph(G_t2.graph['partition'][1])
    B1_B2_tot_weight=sum_weight([B1_G_t2,B2_G_t2])
    G_t1_tot_weight=sum_weight([G_t1])
    print('Total weight of two block in G_t2 before modification',B1_B2_tot_weight,'Original total weight',G_t1_tot_weight)
    return G_t1,G_t2,B1_G_t2,B2_G_t2

def get_positive_edge(G_t1,G_t2):
    # general weight change rate is 1
    # in positive pattern, select edges by determining if F_g2/F_g1 - 1
    positive_edge_G_t1=[]
    positive_edge_G_t2=[]
    G_sum_weight = sum_weight([G_t1, G_t2])
    for edge1 in tqdm(G_t1.edges()):
        if G_t1[edge1[0]][edge1[1]]['weight']==0:
            positive_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
            positive_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])
        else:
            edge_change_rate=G_t2[edge1[0]][edge1[1]]['weight']/G_t1[edge1[0]][edge1[1]]['weight']-1#G_sum_weight[1]/G_sum_weight[0]
            if edge_change_rate>0:
                positive_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
                positive_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])

    G_p1=nx.Graph()
    G_p2=nx.Graph()

    G_p1.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in positive_edge_G_t1])
    G_p2.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in positive_edge_G_t2])
    return G_p1,G_p2


def get_negative_edge(G_t1,G_t2):

    negative_edge_G_t1=[]
    negative_edge_G_t2=[]
    G_sum_weight = sum_weight([G_t1, G_t2])
    for edge1 in tqdm(G_t1.edges()):
        if G_t1[edge1[0]][edge1[1]]['weight']!=0:
            

            edge_change_rate=G_t2[edge1[0]][edge1[1]]['weight']/G_t1[edge1[0]][edge1[1]]['weight']-1 
            if edge_change_rate<0:
                negative_edge_G_t1.append([edge1[0],edge1[1],G_t1[edge1[0]][edge1[1]]['weight']])
                negative_edge_G_t2.append([edge1[0],edge1[1],G_t2[edge1[0]][edge1[1]]['weight']])

    G_n1=nx.Graph()
    G_n2=nx.Graph()

    G_n1.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in negative_edge_G_t1])
    G_n2.add_weighted_edges_from([(edge[0],edge[1],edge[2]) for edge in negative_edge_G_t2])
    return G_n1,G_n2

def flow_boxplot_edge_Q(com_dict_min_best_p,G_p1,G_p2,com_is_list=True):
    #G_p1/G_p2
    result_partition=0
    if com_is_list:
        result_partition=com_dict2list(com_dict_min_best_p)
    else:
        result_partition=com_dict_min_best_p
    G_p1_degree=nx.degree(G_p1,weight='weight')
    G_p2_degree=nx.degree(G_p2,weight='weight')
    tot_weight_g1_g2=sum_weight([G_p1,G_p2])
    Flow_q=[]
    for node_set in result_partition:
        com=G_p1.subgraph(node_set)
        f_q=[]
        for edge in com.edges():
            node=edge[0]
            connected_node=edge[1]
            R_f=G_p1[node][connected_node]['weight']/G_p2[node][connected_node]['weight']
            R_n=(G_p1_degree[node]*G_p1_degree[connected_node]/tot_weight_g1_g2[0])/(G_p2_degree[node]*G_p2_degree[connected_node]/tot_weight_g1_g2[1])

            f_q.append(R_f-R_n)
        Flow_q.append(f_q)

    plt.figure(figsize=(5,5))
    plt.boxplot(Flow_q)
    for i, d in enumerate(Flow_q):
        y = Flow_q[i]
        x = np.random.normal(i + 1, 0.04, len(y))
        plt.scatter(x, y)
    plt.show()
    return Flow_q

def flow_boxplot_edge_ratio(com_dict_min_best_p,G_p1,G_p2,com_is_list=True):
    #G_p1/G_p2
    result_partition=0
    if com_is_list:
        result_partition=com_dict2list(com_dict_min_best_p)
    else:
        result_partition=com_dict_min_best_p
        
    Flow_ratio=[]
    for node_set in result_partition:
        com=G_p1.subgraph(node_set)
        f_ratio=[]
        for edge in com.edges():
            f_ratio.append(G_p2[edge[0]][edge[1]]['weight']/G_p1[edge[0]][edge[1]]['weight'])
        Flow_ratio.append(f_ratio)
        
    plt.figure(figsize=(5,5))
    plt.boxplot(Flow_q)
    for i, d in enumerate(Flow_q):
        y = Flow_q[i]
        x = np.random.normal(i + 1, 0.04, len(y))
        plt.scatter(x, y)
    plt.show()
    
    return Flow_ratio