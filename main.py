import numpy
import random
import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
from collections import *


def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g


def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)


def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))


def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.

    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.

    Arguments:
    num_nodes -- The number of nodes in the returned graph.

    Returns:
    A complete graph in dictionary form.
    """
    result = {}

    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key].add(node_value)

    return result


def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns:
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result


def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals


def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)


def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data) - len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)


def compute_largest_cc_size(g: dict) -> int:
    # Your code here...
    v = {}

    for j in g.keys():
        v[j] = False

    S = 0

    for j in g.keys():
        c = 0
        if not v[j]:
            v[j] = True
            c += 1
            Q = deque()
            Q.append(j)
            while (len(Q) != 0):
                f = Q.popleft()
                for h in g[f]:
                    if not v[h]:
                        v[h] = True
                        c += 1
                        Q.append(h)
            if c > S:
                S = c

    return S


# Graph of actual ISP Service Provider
# 1347 nodes, 3112 edges
isp_graph_random = read_graph("rf7.repr")
isp_graph_targeted = copy_graph(isp_graph_random)


def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u + 1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g


# Generates UPA and ER graphs and copies them
upa_graph_random = upa(1347, 2)
upa_graph_targeted = copy_graph(upa_graph_random)
er_graph_random = erdos_renyi(1347, 0.003432866609)
er_graph_targeted = copy_graph(er_graph_random)


def node_deletion(g, i):
    """
    Given an adjacency list representation of graph g, remove all instances of node i.
    Args:
        g: An undirected graph g = (V,E). A dictionary where nodes are mapped to a set of their neighbors.
        i: A node to be removed

    Returns:
    The adjacency list sans i.
    """
    # Will hold the updated graph
    new_g = {}

    # Remove the entry that has i has its key, and remove i from all values that contain it
    for node in g.keys():
        if node != i:
            new_neighbors = set()
            for neighbor in g[node]:
                if neighbor != i:
                    new_neighbors.add(neighbor)
            new_g[node] = new_neighbors

    return new_g


# Testing node deletion case 1
# g1 = {"a": {"b", "d"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"a", "c"}, "e": {"f"},
#  "f": {"e", "g"}, "g": {"f"}, "h": set(), "i": set()}
# print(node_deletion(g1, "a"))

# Testing node deletion case 2
# Test Case 2 (Expected: 1; Received: 1)
# g2 = {"a": set(), "b": set(), "c": set()}
# print(node_deletion(g1, "a"))

def random_attack_experiment(g, percent_removal):
    """
    Simulates a random attack on a graph by removing nodes randomly
    and computing the size of the largest connected component after each
    node is removed, up to 20% of the total nodes.
    Args:
        g: An directed graph represented as an adjacency list
        percent_removal: Percentage of nodes removed. In this experiment, it's 20%.

    Returns:
    A dictionary containing the size of the largest connected component in g after each node is removed.
    The key corresponds to how many nodes have been removed, and the value is the size of the connected component
    at that stage.
    """
    cc_sizes = {0: compute_largest_cc_size(g)}

    # Find the number of total nodes and find 20% of that
    nodes_to_remove = math.floor(len(g) * percent_removal)
    # Iterate a number of times equal to the counter
    for itr in range(nodes_to_remove):
        # For each iteration, randomly pick a node and remove it
        removed_node = random.choice(list(g.keys()))
        # print(removed_node)
        g = node_deletion(g, removed_node)
        # Compute the largest CC and store that value
        curr_biggest_cc = compute_largest_cc_size(g)
        cc_sizes[itr + 1] = curr_biggest_cc

    return cc_sizes


# Testing random_attack_experiment() case 1
# g1 = {"a": {"b", "d"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"a", "c"}, "e": {"f"}, "f": {"e", "g"}, "g": {"f"},
#        "h": set(), "i": set()}
# print(random_attack_experiment(g1, 1))
# Testing random_attack_experiment() case 2
# g2 = {"a": {"b", "d"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"a", "c"}, "e": {"f"}, "f": {"e", "g"}, "g": {"f"},
#        "h": set(), "i": set()}
# print(random_attack_experiment(g2, 0.5))

def targeted_attack_experiment(g, percent_removal):
    """
    Simulates a targeted attack on a graph by removing nodes in decreasing order of degree
    and computing the size of the largest connected component after each
    node is removed, up to 20% of the total nodes.
    Args:
        g: An directed graph represented as an adjacency list
        percent_removal: Percentage of total nodes to be removed. In this experiment, it's 20%.

    Returns:
    A dictionary containing the size of the largest connected component in g after each node is removed.
    The key corresponds to how many nodes have been removed, and the value is the size of the connected component
    at that stage.
    """
    # Store the initial largest CC
    cc_sizes = {0: compute_largest_cc_size(g)}

    # Find the number of total nodes and find the number to remove
    nodes_to_remove = math.floor(len(g) * percent_removal)
    # Iterate a number of times equal to the counter
    for itr in range(nodes_to_remove):
        # For each iteration, find the node with the highest degree and remove it
        removed_node = ""
        top_degree = -1
        for node in g.keys():
            node_degree = len(g[node])
            if node_degree > top_degree:
                removed_node = node
                top_degree = node_degree
        # print(removed_node)
        g = node_deletion(g, removed_node)
        # Compute the largest CC and store that value
        curr_biggest_cc = compute_largest_cc_size(g)
        cc_sizes[itr + 1] = curr_biggest_cc

    return cc_sizes


# Testing targeted_attack_experiment() case 1
# g1 = {"a": {"b", "d"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"a", "c"}, "e": {"f"}, "f": {"e", "g"}, "g": {"f"},
#       "h": set(), "i": set()}
# print(targeted_attack_experiment(g1, 1))
# Testing targeted_attack_experiment() case 2
# g2 = {"a": {"b", "d"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"a", "c"}, "e": {"f"}, "f": {"e", "g"}, "g": {"f"},
#       "h": set(), "i": set()}
# print(targeted_attack_experiment(g2, 0.5))

# Data from the experiments
isp_random_data = random_attack_experiment(isp_graph_random, 0.2)
isp_targeted_data = targeted_attack_experiment(isp_graph_targeted, 0.2)
upa_random_data = random_attack_experiment(upa_graph_random, 0.2)
upa_targeted_data = targeted_attack_experiment(upa_graph_targeted, 0.2)
er_random_data = random_attack_experiment(er_graph_random, 0.2)
er_targeted_data = targeted_attack_experiment(er_graph_targeted, 0.2)

# Plotting the data
data_list = [isp_random_data,
             isp_targeted_data,
             upa_random_data,
             upa_targeted_data,
             er_random_data,
             er_targeted_data
             ]
graph_title = "Network Resiliency of Graph Topologies to Random and Targeted Attacks"
x_axis = "Nodes Removed"
y_axis = "Largest Connected Component (number of nodes)"
line_labels = ["ISP-Random",
               "ISP-Targeted",
               "UPAG-Random",
               "UPAG-Targeted",
               "ER-Random",
               "ER-Targeted"
               ]
isp_random_graph = plot_lines(data_list, graph_title, x_axis, y_axis, line_labels)
