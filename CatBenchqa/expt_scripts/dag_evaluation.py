import os
import ast
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from collections import defaultdict, deque

class EvaluateDAG():
    def __init__(self, result_csv):
        self.result_csv = result_csv
        self.result_csv['steps'] = self.result_csv['steps'].apply(ast.literal_eval)
        self.total_dags = 0
        self.correct_dags = 0
        self.dag_accuracy = 0
        
        # debugging variables
        self.same_order_correct = 0
        self.same_topo_order_correct = 0
        self.length_mismatch_incorrect = 0
        self.node_missing_incorrect = 0
        self.node_order_incorrect = 0

    def topological_sort(self, num_nodes, edges):
            
        adj = defaultdict(list)
        in_degree = {}
        for u, v in edges:
            in_degree[u] = in_degree.get(u, 0)
            in_degree[v] = in_degree.get(v, 0) + 1
            
        for u, v in edges:
            adj[u].append(v)
            
        # Queue to store vertices with indegree 0
        q = deque()
        for key in in_degree.keys():
            if in_degree[key] == 0:
                q.append(key)
        result = []
        while q:
            node = q.popleft()
            result.append(node)
            # Decrease indegree of adjacent vertices as the current node is in topological order
            for adjacent in adj[node]:
                in_degree[adjacent] -= 1
                # If indegree becomes 0, push it to the queue
                if in_degree[adjacent] == 0:
                    q.append(adjacent)

        # Check for cycle
        if len(result) != num_nodes:
            # print("Graph contains cycle!")
            return []
        return result
    
    def validate_topological_order(self, order, edges, recipe_id=None):
        pos = {node: idx for idx, node in enumerate(order)}
        for u, v in edges:
            if u in pos and v in pos:
                if pos[u] >= pos[v]:
                    self.node_order_incorrect += 1
                    return False
            # disabled so we only check the order for nodes that exist else we don't have to worry since not all questions are in the dataset
            # else: 
            #     self.node_missing_incorrect += 1
            #     return False
        self.same_topo_order_correct += 1
        return True

    def are_equivalent(self, order_a, order_b, edges, recipe_id=None):
        """
        Example to evaluate equivalence of two topological orders of a DAG:
            Input:
                edges = [(5,2), (5,0), (4,0), (4,1), (2,3), (3,1)] #constraints
                order_a = [5, 4, 2, 3, 1, 0]
                order_b = [4, 5, 0, 2, 3, 1]
            Output:
                True
        """
        if order_a == order_b:
            self.same_order_correct += 1
            return True
        if len(order_a) != len(order_b):
            self.length_mismatch_incorrect += 1
            return False
        return (self.validate_topological_order(order_a, edges, recipe_id) and 
                self.validate_topological_order(order_b, edges, recipe_id))
        
    def get_num_nodes(self, edges):
        unique = set([num for edge in edges for num in edge])
        return len(unique)
    
    def get_true_edges(self, recipe_id):
        edges = []
        for k in range(len(self.result_csv)):
            if self.result_csv.iloc[k]['recipe_name'] == recipe_id and self.result_csv.iloc[k]['label'] == 1:
                steps = self.result_csv.iloc[k]['steps']
                enumerated_steps = list(enumerate(steps, start=1))
                sentence1 = self.result_csv.iloc[k]['sentence1']
                sentence2 = self.result_csv.iloc[k]['sentence2']
                i = next((step[0] for step in enumerated_steps if step[1] == sentence1), None)
                j = next((step[0] for step in enumerated_steps if step[1] == sentence2), None)
                edges.append([j, i])
                
        return edges
                  
    def get_pred_edges(self, recipe_id):
        edges = []
        for k in range(len(self.result_csv)):
            if self.result_csv.iloc[k]['recipe_name'] == recipe_id and self.result_csv.iloc[k]['prediction'] == 1:
                steps = self.result_csv.iloc[k]['steps']
                enumerated_steps = list(enumerate(steps, start=1))
                sentence1 = self.result_csv.iloc[k]['sentence1']
                sentence2 = self.result_csv.iloc[k]['sentence2']
                i = next((step[0] for step in enumerated_steps if step[1] == sentence1), None)
                j = next((step[0] for step in enumerated_steps if step[1] == sentence2), None)
                edges.append((j, i))
                
        return edges
    
    def get_recipe_dag(self, recipe_id):
        true_edges = self.get_true_edges(recipe_id)
        num_nodes = self.get_num_nodes(true_edges)
        return self.topological_sort(num_nodes, true_edges)
    
    def get_pred_dag(self, recipe_id):
        pred_edges = self.get_pred_edges(recipe_id)
        num_nodes = self.get_num_nodes(pred_edges)
        return self.topological_sort(num_nodes, pred_edges)
    
    def get_dag_accuracy(self):
        recipe_hashmap = {}
        for i in range(len(self.result_csv)):
            recipe_id = self.result_csv.iloc[i]['recipe_name']
            if recipe_id not in recipe_hashmap:
                recipe_hashmap[recipe_id] = {}
            else:
                continue
            
            recipe_hashmap[recipe_id]['true_edges'] = self.get_true_edges(recipe_id)
            recipe_hashmap[recipe_id]['num_nodes'] = self.get_num_nodes(recipe_hashmap[recipe_id]['true_edges'])
            recipe_dag = self.topological_sort(recipe_hashmap[recipe_id]['num_nodes'], recipe_hashmap[recipe_id]['true_edges'])
            recipe_hashmap[recipe_id]['true_order'] = recipe_dag
            
            recipe_hashmap[recipe_id]['predicted_edges'] = self.get_pred_edges(recipe_id)
            pred_dag = self.topological_sort(recipe_hashmap[recipe_id]['num_nodes'], recipe_hashmap[recipe_id]['predicted_edges'])
            recipe_hashmap[recipe_id]['pred_order'] = pred_dag
            
            recipe_hashmap[recipe_id]['equivalent'] = self.are_equivalent(recipe_dag, pred_dag, recipe_hashmap[recipe_id]['true_edges'], recipe_id)
            if recipe_hashmap[recipe_id]['equivalent']:
                self.correct_dags += 1
            self.total_dags += 1
            
        self.dag_accuracy = self.correct_dags / self.total_dags
        print("[DAG] Accuracy:", self.dag_accuracy)
        print("[DAG] Total dags:", self.total_dags)
        print("-----------------------------------")
        print("[DAG] Same order correct:", self.same_order_correct)
        print("[DAG] Same topological order correct:", self.same_topo_order_correct)
        print("-----------------------------------")
        print("[DAG] Cycles deteceted incorrect:", self.length_mismatch_incorrect)
        # print("[DAG] Node missing incorrect:", self.node_missing_incorrect)
        print("[DAG] Node order incorrect:", self.node_order_incorrect)
        
    def remove_redundant_edges(self, edges):
        """
        If there is an edge from u to v and there is an edge from v to w, then the edge from u to w is redundant.
        """
        true_edges = []
        for i in range(len(edges)):
            u, v = edges[i]
            def find_all_paths(graph, u, v):
                all_paths = list(nx.all_simple_paths(graph, u, v))
                return all_paths
            all_u_v_paths = find_all_paths(nx.DiGraph(edges), u, v)
            if len(all_u_v_paths) == 1:
                true_edges.append([u, v])
        return true_edges
    
    def visualize_dags(self, recipe_id, save_path):
        G_true = nx.DiGraph()
        G_pred = nx.DiGraph()
        
        edges_true = self.get_true_edges(recipe_id)
        # edges_true = self.remove_redundant_edges(edges_true)
        edges_true = [(str(u), str(v)) for u, v in edges_true]  # Convert to list of tuples
        
        G_true.add_edges_from(edges_true)
        
        edges_pred = self.get_pred_edges(recipe_id)
        edges_pred = self.remove_redundant_edges(edges_pred)
        edges_pred = [(str(u), str(v)) for u, v in edges_pred]  # Convert to list of tuples
        G_pred.add_edges_from(edges_pred)
        
        true_dag = self.get_recipe_dag(recipe_id)
        pred_dag = self.get_pred_dag(recipe_id)
        
        pos_true = nx.spring_layout(G_true)
        pos_pred = nx.spring_layout(G_pred)
        
        plt.figure(figsize=(10, 6))
        nx.draw(G_true, pos_true, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold', arrows=True)
        edge_labels_true = {(u, v): f"{true_dag.index(int(u))}" for (u, v) in G_true.edges()}
        nx.draw_networkx_edge_labels(G_true, pos_true, edge_labels=edge_labels_true)
        plt.title("True DAG with Topological Order")
        plt.axis('off')
        plt.savefig(os.path.join(save_path, "true_dag.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        nx.draw(G_pred, pos_pred, with_labels=True, node_color='lightgreen', node_size=500, font_size=12, font_weight='bold', arrows=True)
        edge_labels_pred = {(u, v): f"{pred_dag.index(int(u))}" for (u, v) in G_pred.edges()}
        nx.draw_networkx_edge_labels(G_pred, pos_pred, edge_labels=edge_labels_pred)
        plt.title("Predicted DAG with Topological Order")
        plt.axis('off')
        plt.savefig(os.path.join(save_path, "pred_dag.png"))
        plt.close()
        
        return true_dag, pred_dag, edges_true
            
    
if __name__ == "__main__":
    
    # Load the dataset
    DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
    binary_label_path = os.path.join(DATASET_PATH, "binary_label")
    
    original_csv = pd.read_csv(os.path.join(binary_label_path, 'eval_test.csv'))
    result_csv = original_csv.copy()
    result_csv['prediction'] = result_csv['label']
    
    evaluator = EvaluateDAG(result_csv)
    # evaluator.get_dag_accuracy()
    
    edges = [[1,2], [1,3], [1,4], [2,3], [3,4], [4,5]]
    print(evaluator.remove_redundant_edges(edges))