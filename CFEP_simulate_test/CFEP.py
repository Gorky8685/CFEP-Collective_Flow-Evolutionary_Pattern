
from __future__ import print_function

import array
from decimal import Decimal

import numbers
import warnings

import networkx as nx
import numpy as np
import random
import time
from numba import jit

from community_status import Status


__PASS_MAX = -1
__MIN = 0.0000001


def check_random_state(seed):

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,after_graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None,
                   obj_func=None, 
                   obj_dir=None,
                   show_process=False):
    
    dendo = generate_dendrogram(graph,after_graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state,
                                obj_func, 
                                obj_dir,
                                show_process)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph,after_graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None,
                        obj_func=None,
                        obj_dir='max',
                        show_process=False):
    
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    after_graph_copy = after_graph.copy()
    
    status = Status()
    status_after = Status()
    
    status.init(current_graph, weight, part_init)
    status_after.init(after_graph_copy, weight, part_init)
    
    status_list = list()
    __one_level(current_graph, after_graph_copy, status, status_after, weight, resolution, random_state, obj_func, obj_dir,show_process)

    
    
    new_mod = __modularity(status,status_after, resolution, obj_func, current_graph, after_graph_copy, weight)

    partition = __renumber(status.node2com)
    partition_after = __renumber(status_after.node2com)
    status_list.append(partition)
    return status.node2com


def induced_graph(partition, graph, weight="weight"):
    
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())
    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        if com1==com2:
            continue
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):

    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):

    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph

def stopping_criteria(stable_Q_list, check_continue_epoch=100):
    if len(stable_Q_list)<10:
        return True
    else:
        last_10=stable_Q_list[check_continue_epoch*-1:]
        last_10=np.array(last_10)
        last_10_TF=last_10==last_10[-1]
        if False not in last_10_TF and last_10[-1]==1:
            return False
        else:
            return True
        
# @jit(nopython=True)
def __one_level(graph, graph_after, status, status_after, weight_key, resolution, random_state, obj_func, obj_dir,show_process):

    modified = True
    nb_pass_done = -2
    cur_mod = __modularity(status, status_after, resolution, obj_func, graph, graph_after,weight_key)
    if show_process:
        print('------------')
        print('in one level')
        print('cur_mod ',cur_mod)

    new_mod = cur_mod
    stable_Q_list=[]
    
    while stopping_criteria(stable_Q_list, check_continue_epoch=10):
#     while nb_pass_done<0:
        t = time.time()
        cur_mod = new_mod
#         modified = False
        nb_pass_done += 1
        if show_process:
            print('--------------')
            print(str(nb_pass_done)+' times iteration')
        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            neigh_communities = __neighcom(node, graph, status, weight_key)
            neigh_communities_after = __neighcom(node, graph_after, status_after, weight_key)

#             best_increase = __modularity(status, status_after, resolution, obj_func, graph, graph_after, weight_key)
            best_increase=0
            best_com = com_node
            
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                before_modified_mod = __modularity_in_neighcom(status,status_after, graph, graph_after, 
                                                                   weight_key,[com,com_node],obj_dir,resolution)
                __remove(node, com_node,
                         neigh_communities.get(com_node, 0.), 
                         neigh_communities_after.get(com_node, 0.), 
                         status, status_after)
                
                __insert(node, com,
                         neigh_communities.get(com, 0.), 
                         neigh_communities_after.get(com, 0.), status, status_after)

                after_modified_mod = __modularity_in_neighcom(status,status_after, graph, graph_after, 
                                                                  weight_key,[com,com_node],obj_dir,resolution)
                modified_increase=after_modified_mod-before_modified_mod
                
                if  modified_increase-best_increase>__MIN: #or modified_increase-best_increase==0
                    best_increase = modified_increase
                    best_com = com


                __remove(node, com,
                     neigh_communities.get(com, 0.), 
                     neigh_communities_after.get(com, 0.), 
                     status, status_after)
                __insert(node, com_node,
                         neigh_communities.get(com_node, 0.), 
                         neigh_communities_after.get(com_node, 0.), 
                         status, status_after)
                    
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), 
                     neigh_communities_after.get(com_node, 0.), 
                     status, status_after)
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), neigh_communities_after.get(best_com, 0.), status, status_after)
            
            if show_process:
                print('node '+str(node),'connected com ', len(neigh_communities),'Q increase ',best_increase,'best com ',best_com)

        
        new_mod = __modularity(status,status_after, resolution, obj_func, graph, graph_after, weight_key)
        if show_process:
            print(new_mod)
            print(f'coast:{time.time() - t:.4f}s')
            print(status.node2com)
        if new_mod - cur_mod < __MIN:
            stable_Q_list.append(1)
        else:
            stable_Q_list.append(0)
            
    # print(stable_Q_list)
        
            
# @jit(nopython=True)
def __modularity_in_neighcom(status,status_after, graph, graph_after, weight_key,com_before_after,obj_dir,resolution):
    modified_mod = 0.    
    for community in com_before_after:
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)

        if len(node_list_in_com)>1:
            for link in graph.subgraph(node_list_in_com).edges(data=True):

                graph_degree=graph.degree(weight=weight_key)[link[0]]*graph.degree(weight=weight_key)[link[1]]
                graph_after_degree=graph_after.degree(weight=weight_key)[link[0]]*graph_after.degree(weight=weight_key)[link[1]]
                if graph_after[link[0]][link[1]][weight_key]>__MIN and graph_after_degree>__MIN:
                    if obj_dir=='max':
                        modified_mod+=graph[link[0]][link[1]][weight_key]/graph_after[link[0]][link[1]][weight_key]-resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight)
                        
                    if obj_dir=='min':
                        modified_mod+=resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight)-graph[link[0]][link[1]][weight_key]/graph_after[link[0]][link[1]][weight_key]
                        
                else:
                    modified_mod+=0
            
    return modified_mod
    
    
def __neighcom(node, graph, status, weight_key):

    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, weight_after, status,status_after):

    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1
    
    status_after.degrees[com] = (status_after.degrees.get(com, 0.)
                           - status_after.gdegrees.get(node, 0.))
    status_after.internals[com] = float(status_after.internals.get(com, 0.) -
                                  weight_after - status_after.loops.get(node, 0.))
    status_after.node2com[node] = -1


def __insert(node, com, weight, weight_after, status, status_after):

    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))
    
    status_after.node2com[node] = com
    status_after.degrees[com] = (status_after.degrees.get(com, 0.) +
                           status_after.gdegrees.get(node, 0.))
    status_after.internals[com] = float(status_after.internals.get(com, 0.) +
                                  weight_after + status_after.loops.get(node, 0.))


def __modularity(status,status_after, resolution ,obj_func, graph, graph_after, weight_key):
    if obj_func=='minus':
        return __modularity_minus(status,status_after, resolution)
    elif obj_func=='fraction_max':
        return __modularity_fraction(status,status_after, resolution, graph, graph_after, weight_key)
    elif obj_func=='fraction_min':
        return __modularity_fraction_min(status,status_after, resolution, graph, graph_after, weight_key)
    elif obj_func=='fraction_add':
        return __modularity_fraction_add(status,status_after, resolution, graph, graph_after, weight_key)
    elif obj_func=='original':
        return __modularity_original(status, resolution)
    else:
        print('The objective function is not in the list.')

def __modularity_fraction(status,status_after, resolution,graph, graph_after, weight_key):

    result = 0.
    for community in set(status.node2com.values()):
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)
                
        internal_fraction=0
        if len(node_list_in_com)>1:
            for link in graph.subgraph(node_list_in_com).edges(data=True):
                if (link[1]!=link[0]):

                    graph_degree=graph.degree(weight=weight_key)[link[0]]*graph.degree(weight=weight_key)[link[1]]
                    graph_after_degree=graph_after.degree(weight=weight_key)[link[0]]*graph_after.degree(weight=weight_key)[link[1]]
                    if graph_after[link[0]][link[1]][weight_key]>__MIN and graph_after_degree>__MIN:

                        result+=graph[link[0]][link[1]][weight_key]/graph_after[link[0]][link[1]][weight_key]-resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight)

                    else:
                        result+=0

    return result

def __modularity_fraction_min(status,status_after, resolution,graph, graph_after, weight_key):

    result = 0.
    for community,community_after in zip(set(status.node2com.values()),set(status_after.node2com.values())):
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)
                
        internal_fraction=0
        if len(node_list_in_com)>1:
            for link in graph.edges(data=True):
                if (link[0] in node_list_in_com) & (link[1] in node_list_in_com) & (link[1]!=link[0]):

                    graph_degree=graph.degree(weight=weight_key)[link[0]]*graph.degree(weight=weight_key)[link[1]]
                    graph_after_degree=graph_after.degree(weight=weight_key)[link[0]]*graph_after.degree(weight=weight_key)[link[1]]
                    if graph_after[link[0]][link[1]][weight_key]>__MIN and graph_after_degree>__MIN:

                        result+=resolution*(graph_degree/graph_after_degree)/(status.total_weight/status_after.total_weight)-graph[link[0]][link[1]][weight_key]/graph_after[link[0]][link[1]][weight_key]

                    else:
                        result+=0

    return result

def __modularity_fraction_add(status,status_after, resolution,graph, graph_after, weight_key):

    print('in fraction model')
    result = 0.
    for community,community_after in zip(set(status.node2com.values()),set(status_after.node2com.values())):
        # find nodes in current community
        node_list_in_com=[]
        for node,corr_com in status.node2com.items():
            if corr_com==community:
                node_list_in_com.append(node)
                
        internal_fraction=0
        if len(node_list_in_com)>1:
            for link in graph.edges(data=True):
                if (link[0] in node_list_in_com) & (link[1] in node_list_in_com) & (link[1]!=link[0]):
                    
                    graph_degree=graph.degree(weight='weight')[link[0]]+graph.degree(weight='weight')[link[1]]
                    graph_after_degree=graph_after.degree(weight='weight')[link[0]]+graph_after.degree(weight='weight')[link[1]]
                    
                    if graph_after[link[0]][link[1]][weight_key]>__MIN and graph_after_degree>__MIN:

                        result+=graph[link[0]][link[1]][weight_key]/graph_after[link[0]][link[1]][weight_key]-(graph_degree)/(graph_after_degree)
                        print([link[0]],[link[1]],graph[link[0]][link[1]][weight_key],graph_after[link[0]][link[1]][weight_key],result)
                        print(graph.degree(weight='weight')[link[0]],graph.degree(weight='weight')[link[1]])
                        print(graph_after.degree(weight='weight')[link[0]],graph_after.degree(weight='weight')[link[1]])
                    else:
                        result+=0
                        
    return result

def __modularity_minus(status,status_after, resolution):

    links = float(status.total_weight)
    result = 0.

    for community,community_after in zip(set(status.node2com.values()),set(status_after.node2com.values())):
        in_degree = status.internals.get(community, 0.)
        degree = status_after.internals.get(community_after, 0.)
        if links > 0:
            result += in_degree  -  degree

    return result

def __modularity_original(status, resolution):

    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = float(status.internals.get(community, 0.))
        degree = float(status.degrees.get(community, 0.))
        if links > 0:
            result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):

    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

def generate_test_graph(sizes, probs, com_m, seed):

    random.seed(seed)

    def _random_subset(seq, m, rng):

        targets = set()
        while len(targets) < m:
            x = random.choice(seq)
            targets.add(x)
        return targets

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    # edit each partition to be scale-free by BA model
    for partition,m in zip(G.graph['partition'],com_m):
        partition=list(partition)
        n=len(partition)
        initial_graph=None

        if m < 1 or m >= n:
            raise nx.NetworkXError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )
        
        star_graph_nodes=random.sample(partition,m+1) #select m+1 nodes to build star graph
        
        # Select the remaining nodes after building the star network for later addition to the star network
        rest_node=partition.copy()
        for node in star_graph_nodes:
            rest_node.remove(node)

        hub=random.choice(star_graph_nodes)
        star_graph_nodes.remove(hub)

        # build star graph
        G.add_edges_from([(hub,connected_node) for connected_node in star_graph_nodes])
        G_star=nx.Graph()
        G_star.add_edges_from([(hub,connected_node) for connected_node in star_graph_nodes])

        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = [n for n, d in G_star.degree(partition) for _ in range(d)]

        # Start adding the other n - m nodes.
        for source in rest_node:
            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)
            targets = _random_subset(repeated_nodes, m, seed)
            # Add edges to m nodes from the source.
            G.add_edges_from(zip([source] * m, targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)
            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source] * m)
    return G