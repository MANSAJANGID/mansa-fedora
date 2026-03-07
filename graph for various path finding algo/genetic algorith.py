import random

class GraphGeneticPathfinder:
    def __init__(self, graph, start, goal, population_size=100, generations=200, mutation_rate=0.1):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.pop_size = population_size
        self.gens = generations
        self.mutation_rate = mutation_rate
        # Max path length to prevent infinite loops (number of nodes in graph)
        self.max_path_length = len(graph)

    def create_individual(self):
        """Creates a random valid path from start to whenever it gets stuck or reaches goal."""
        path = [self.start]
        current = self.start
        
        for _ in range(self.max_path_length):
            neighbors = list(self.graph[current].keys())
            if not neighbors or current == self.goal:
                break
            current = random.choice(neighbors)
            path.append(current)
            if current == self.goal:
                break
        return path

    def calculate_fitness(self, path):
        """Scores a path. Lower total edge weight is better."""
        if not path or path[0] != self.start:
            return 0
        
        current_node = path[0]
        total_cost = 0
        valid_steps = 0
        
        # Traverse the path and validate edges
        for i in range(1, len(path)):
            next_node = path[i]
            if next_node in self.graph[current_node]:
                total_cost += self.graph[current_node][next_node]
                current_node = next_node
                valid_steps += 1
                if current_node == self.goal:
                    # Huge reward for reaching goal, penalized by cost
                    return 10000 / (total_cost + 1)
            else:
                # Broken path penalty
                break
        
        # If goal not reached, fitness is based on distance to goal
        # Since we don't have coordinates, we use a large penalty
        return 1 / (total_cost + 100)

    def crossover(self, parent1, parent2):
        """Finds a common node in both paths and swaps the tails."""
        # Find common nodes (excluding start)
        common_nodes = list(set(parent1[1:-1]) & set(parent2[1:-1]))
        
        if not common_nodes:
            # If no common nodes, return parents as is
            return parent1[:] if random.random() > 0.5 else parent2[:]
        
        node = random.choice(common_nodes)
        idx1 = parent1.index(node)
        idx2 = parent2.index(node)
        
        return parent1[:idx1] + parent2[idx2:]

    def mutate(self, path):
        """Randomly picks a point and re-generates the rest of the path."""
        if len(path) < 2 or random.random() > self.mutation_rate:
            return path
        
        mutation_point = random.randint(0, len(path) - 1)
        new_path = path[:mutation_point + 1]
        current = new_path[-1]
        
        for _ in range(self.max_path_length - len(new_path)):
            neighbors = list(self.graph[current].keys())
            if not neighbors or current == self.goal:
                break
            current = random.choice(neighbors)
            new_path.append(current)
            if current == self.goal:
                break
        return new_path

    def run(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        for gen in range(self.gens):
            population.sort(key=lambda x: self.calculate_fitness(x), reverse=True)
            
            if gen % 50 == 0:
                print(f"Generation {gen}: Best Path Cost = {10000/self.calculate_fitness(population[0]) - 1:.2f}")

            # Elitism: Keep top 10%
            elite_count = max(2, self.pop_size // 10)
            next_gen = population[:elite_count]
            
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(population[:self.pop_size//2], 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)
            
            population = next_gen

        return population[0]

# --- Main Execution ---

# {Node: {Neighbor: Cost}}
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'C': 2, 'D': 8},
    'C': {'A': 5, 'B': 2, 'D': 1},
    'D': {'B': 8, 'C': 1, 'E': 3},
    'E': {'D': 3}
}

print("--- Graph Connections ---")
for node, neighbors in graph.items():
    print(f"{node}: {neighbors}")

valid_nodes = list(graph.keys())

try:
    print(f"\nAvailable Nodes: {valid_nodes}")
    start_node = input("Enter Start Node: ").strip().upper()
    goal_node = input("Enter Goal Node: ").strip().upper()

    if start_node not in graph or goal_node not in graph:
        print("Error: Node not found in graph.")
    else:
        print("\nEvolving paths...")
        gp = GraphGeneticPathfinder(graph, start_node, goal_node)
        best_path = gp.run()
        
        print("\nFinal Result:")
        print(f"Path: {' -> '.join(best_path)}")
        print(f"Fitness Score: {gp.calculate_fitness(best_path):.4f}")

except (ValueError, KeyError) as e:
    print(f"Invalid input. Error: {e}")
