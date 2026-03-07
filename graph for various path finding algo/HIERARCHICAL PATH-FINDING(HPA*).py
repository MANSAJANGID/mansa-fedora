import heapq

class GraphAStar:
    def __init__(self, graph, heuristics=None):
        """
        Initializes the Graph A* solver.
        graph: Adjacency list format {node: {neighbor: edge_cost}}
        heuristics: Dictionary of estimated costs to the goal {node: cost}
        """
        self.graph = graph
        self.heuristics = heuristics if heuristics else {}

    def heuristic(self, node, goal):
        # In a graph without physical coordinates, we rely on a lookup table.
        # If no heuristic is provided, it returns 0 (behaving exactly like Dijkstra's Algorithm).
        return self.heuristics.get(node, 0)

    def get_neighbors(self, node):
        # Yields (neighbor, edge_cost) directly from the graph dictionary
        if node in self.graph:
            for neighbor, cost in self.graph[node].items():
                yield neighbor, cost

    def solve(self, start, goal):
        # Priority Queue for A* search: (priority, current_node)
        pq = [(0, start)]
        g_score = {start: 0}
        parent = {start: None}
        
        while pq:
            _, current = heapq.heappop(pq)
            
            # Goal Reached
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1], g_score[goal]
            
            # Explore Neighbors
            for neighbor, cost in self.get_neighbors(current):
                new_g = g_score[current] + cost
                
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    priority = new_g + self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (priority, neighbor))
                    parent[neighbor] = current
                    
        return None, float('inf')

# --- Main Execution Environment ---

def main():
    # 1. Define the Graph (Adjacency List)
    graph = {
        'A': {'B': 2, 'C': 5, 'D': 10},
        'B': {'A': 2, 'C': 2, 'E': 4},
        'C': {'A': 5, 'B': 2, 'D': 1, 'GOAL': 8},
        'D': {'A': 10, 'C': 1, 'F': 2},
        'E': {'B': 4, 'GOAL': 3},
        'F': {'D': 2, 'GOAL': 6},
        'GOAL': {}
    }

    # 2. Define Heuristics (Estimated cost to 'GOAL')
    heuristics = {
        'A': 7,
        'B': 5,
        'C': 4,
        'D': 3,
        'E': 2,
        'F': 2,
        'GOAL': 0
    }

    print("=" * 50)
    print("A* Search - Graph Pathfinding")
    print("=" * 50)
    print("\nGraph Connections & Edge Costs:")
    for node, edges in graph.items():
        print(f"Node {node} -> {edges}")

    valid_nodes = list(graph.keys())

    try:
        print(f"\nAvailable Nodes: {valid_nodes}")
        
        # User Input Handling
        while True:
            start_node = input("\nEnter Start Node: ").strip().upper()
            if start_node in valid_nodes: break
            print(f"Invalid. Choose from {valid_nodes}")
            
        while True:
            goal_node = input("Enter Goal Node: ").strip().upper()
            if goal_node in valid_nodes: break
            print(f"Invalid. Choose from {valid_nodes}")

        if start_node == goal_node:
            print("Start and Goal are the same node!")
            return

        # Initialize Solver and Run
        solver = GraphAStar(graph, heuristics)
        path, cost = solver.solve(start_node, goal_node)
        
        print("\n" + "=" * 50)
        if path:
            print(f"SUCCESS: Path found!")
            print(f"Total Cost: {cost}")
            print(f"Path sequence: {' -> '.join(path)}")
        else:
            print("FAILED: No valid path exists between these nodes.")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
