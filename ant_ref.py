from utils import Undirected_graph, TEMPLATE_GRAPH_PATH

class Ant_Ref(Undirected_graph):
    def __init__(self, graph_name: str = "ant_ref", node_num: int = 5, edge_prob: float = 0.5):
        super().__init__(graph_name, node_num, edge_prob)
        self. 
        



 def main():
     agent = Ant_Ref()
     agent.load_graph(TEMPLATE_GRAPH_PATH)

if __name__ == "__main__":
    main()