"""
template https://snap.stanford.edu/snappy/doc/tutorial/tutorial.html#basic-types

G1 = snap.TUNGraph.New()
G1.AddNode(1)
G1.AddNode(5)
G1.AddNode(32)


G1.AddEdge(1,5)
G1.AddEdge(5,1)
G1.AddEdge(5,32)

G2 = snap.GenRndGnm(snap.TUNGraph, 20, 50)

import networkx as nx
import matplotlib.pyplot as plt

# 将Snap图转换为NetworkX图
G = nx.Graph()
for EI in G2.Edges():
    G.add_edge(EI.GetSrcNId(), EI.GetDstNId())

# 绘制图
nx.draw(G, with_labels=True)
plt.show()

"""
"""
思路， 针对于每一个顶点 一个improve_payoff 方法

遍历顶点
improve_payoff
check_nash_equilibrium 
    -> nash  -> end
-> next vertex

"""
import snap
import networkx as nx
import matplotlib.pyplot as plt

def import_template_graph():
    file_path = "src/graph.txt"
    G = snap.LoadEdgeList(snap.PUNGraph, file_path, 0, 1)
    return G

def draw_graph(graph):

    G = nx.Graph()
    for EI in graph.Edges():
        G.add_edge(EI.GetSrcNId(), EI.GetDstNId())

    nx.draw(G, with_labels=True)
    plt.show()

def main():
    G = import_template_graph()
    draw_graph(G)

if __name__ == '__main__':
    main()