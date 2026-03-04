import random
from collections import deque

class GraphAntColonyOptimizer:
    def __init__(self, graph, start, goal, num_ants=10, iterations=50, evaporation=0.5, alpha=1.0, beta=2.0):
        """
        Initialize the Graph Ant Colony Optimizer.
        
        Args:
            graph: Dictionary representing adjacency list {node: {neighbor: cost}}
            start: Starting node identifier
            goal: Target node identifier
            num_ants: Number of ants per iteration
            iterations: Number of iterations to run
            evaporation: Pheromone evaporation rate (0-1)
            alpha: Weight of pheromone importance
            beta: Weight of heuristic importance (edge cost)
        """
        self.graph = graph
        self.start = start
        self.goal = goal
        self.num_ants = num_ants
        self.iterations = iterations
        self.evaporation = max(0, min(evaporation, 1))
        self.alpha = alpha
        self.beta = beta
        
        # Initialize pheromones to 1.0 for every directed edge in the graph
        self.pheromones = {}
        for u in self.graph:
            for v in self.graph[u]:
                self.pheromones[(u, v)] = 1.0

    def is_path_exists(self):
        """Check if a valid path exists using BFS."""
        if self.start not in self.graph or self.goal not in self.graph:
            return False
        
        queue = deque([self.start])
        visited = {self.start}
        
        while queue:
            current = queue.popleft()
            if current == self.goal:
                return True
            
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False

    def get_path_cost(self, path):
        """Calculates the total weight/cost of a given path."""
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i+1]]
        return cost

    def run(self):
        """Run the ACO algorithm and return the best path found."""
        if not self.is_path_exists():
            print("ERROR: No valid path exists from start to goal in this graph!")
            return None, float('inf')
        
        best_path = None
        min_cost = float('inf')

        for i in range(self.iterations):
            paths = []
            for _ in range(self.num_ants):
                path = self.simulate_ant()
                if path:
                    paths.append(path)
                    current_cost = self.get_path_cost(path)
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_path = path

            self.update_pheromones(paths)
            
            if i % 10 == 0:
                status = f"{min_cost} cost" if min_cost != float('inf') else "not found"
                print(f"Iteration {i}: Best path = {status}")

        return best_path, min_cost

    def simulate_ant(self):
        """Simulate one ant's journey through the graph."""
        current = self.start
        path = [current]
        visited = {current}
        max_steps = len(self.graph) * 2 # Safety break for cyclic graphs

        for _ in range(max_steps):
            if current == self.goal:
                return path
            
            # Get unvisited neighbors
            neighbors = [n for n in self.graph[current] if n not in visited]
            
            if not neighbors:
                break # Ant reached a dead end

            probs = []
            for n in neighbors:
                # tau = Pheromone level on the edge
                tau = self.pheromones[(current, n)] ** self.alpha
                # eta = Inverse of the edge cost (heuristic)
                eta = (1.0 / self.graph[current][n]) ** self.beta
                probs.append(tau * eta)

            total = sum(probs)
            if total == 0:
                next_node = random.choice(neighbors)
            else:
                probs = [p / total for p in probs]
                next_node = random.choices(neighbors, weights=probs)[0]
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return None # Return None if the ant failed to reach the goal

    def update_pheromones(self, paths):
        """Update pheromone levels on edges based on successful paths."""
        # 1. Evaporate all edges
        for edge in self.pheromones:
            self.pheromones[edge] *= (1 - self.evaporation)
            self.pheromones[edge] = max(self.pheromones[edge], 0.0001) # Prevent absolute zero
        
        # 2. Deposit new pheromones
        for path in paths:
            path_cost = self.get_path_cost(path)
            # More pheromone deposited for shorter/cheaper paths
            deposit = 10.0 / path_cost 
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                self.pheromones[(u, v)] += deposit


def main():
    # Define a graph as an Adjacency List: {Node: {Neighbor: Edge_Weight}}
    graph = {
        'A': {'B': 2, 'C': 5, 'D': 10},
        'B': {'A': 2, 'C': 2, 'E': 4},
        'C': {'A': 5, 'B': 2, 'D': 1, 'E': 3},
        'D': {'A': 10, 'C': 1, 'F': 2},
        'E': {'B': 4, 'C': 3, 'F': 1},
        'F': {'D': 2, 'E': 1}
    }

    print("=" * 50)
    print("Ant Colony Optimization - Graph Pathfinding")
    print("=" * 50)
    print("\nGraph Connections & Edge Costs:")
    for node, edges in graph.items():
        print(f"Node {node} -> {edges}")

    valid_nodes = list(graph.keys())

    try:
        print(f"\nValid Nodes: {valid_nodes}")
        
        while True:
            start_node = input("\nEnter Start Node: ").strip().upper()
            if start_node in valid_nodes:
                break
            print(f"Invalid. Please choose from {valid_nodes}")
        
        while True:
            goal_node = input("Enter Goal Node: ").strip().upper()
            if goal_node in valid_nodes:
                break
            print(f"Invalid. Please choose from {valid_nodes}")

        if start_node == goal_node:
            print(f"Start and goal are the same: {start_node}")
            return

        print(f"\nStart: {start_node}")
        print(f"Goal: {goal_node}")
        print("\nAnts are exploring the graph...\n")

        # Initialize and run ACO
        aco = GraphAntColonyOptimizer(graph, start_node, goal_node, num_ants=20, iterations=100)
        final_path, total_cost = aco.run()

        print("\n" + "=" * 50)
        if final_path:
            print(f"SUCCESS: Optimal Path Found!")
            print(f"Total Cost: {total_cost}")
            print(f"Path: {' -> '.join(final_path)}")
        else:
            print("FAILED: No path found by the colony.")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
