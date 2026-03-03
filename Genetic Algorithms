import random

class GeneticPathfinder:
    def __init__(self, grid, start, goal, population_size=100, generations=200):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = tuple(start)  # Convert to tuple for consistency
        self.goal = tuple(goal)    # Convert to tuple for consistency
        self.pop_size = population_size
        self.gens = generations
        # More reasonable max moves (optimal path length upper bound)
        self.max_moves = (self.rows + self.cols) * 2
        # Moves: 0:Up, 1:Down, 2:Left, 3:Right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def create_individual(self):
        """Creates a random sequence of moves."""
        return [random.randint(0, 3) for _ in range(self.max_moves)]

    def calculate_fitness(self, individual):
        """Scores an individual. Higher is better."""
        curr = list(self.start)
        moves_taken = 0
        
        for move_idx in individual:
            dr, dc = self.directions[move_idx]
            nr, nc = curr[0] + dr, curr[1] + dc;
            
            # If move is valid and not a wall, update position
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                curr = [nr, nc]
                moves_taken += 1
                # Check if goal reached
                if tuple(curr) == self.goal:
                    # Reward for reaching goal with fewer moves
                    return 1000 / (moves_taken + 1)
            
            # Stop if too many moves attempted
            if moves_taken > self.max_moves * 2:
                break;
        
        # Distance from current end point to the actual goal
        dist_to_goal = abs(curr[0] - self.goal[0]) + abs(curr[1] - self.goal[1])
        
        # Fitness: Inverse of distance (closer = better)
        return 1 / (dist_to_goal + 1)

    def crossover(self, parent1, parent2):
        """Combines two parents to create a child."""
        point = random.randint(1, self.max_moves - 1)
        return parent1[:point] + parent2[point:]

    def mutate(self, individual, rate=0.05):
        """Randomly changes moves to maintain diversity."""
        individual = individual[:]  # Make a copy to avoid modifying original
        for i in range(len(individual)):
            if random.random() < rate:
                individual[i] = random.randint(0, 3)
        return individual

    def run(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        
        for gen in range(self.gens):
            # Sort population by fitness
            population.sort(key=lambda x: self.calculate_fitness(x), reverse=True)
            
            # Print progress every 50 generations
            if gen % 50 == 0:
                best_fit = self.calculate_fitness(population[0])
                print(f"Generation {gen}: Best Fitness = {best_fit:.4f}")
            
            # Check if goal reached
            if self.calculate_fitness(population[0]) > 900:
                print(f"Goal reached at generation {gen}!")
                break;

            # Selection (Take top 20% to breed)
            elite = population[:max(2, self.pop_size // 5)]  # Ensure at least 2 for breeding
            next_gen = elite[:]
            
            while len(next_gen) < self.pop_size:
                if len(elite) >= 2:
                    p1, p2 = random.sample(elite, 2)
                else:
                    p1, p2 = elite[0], elite[0]
                    
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)
            
            population = next_gen

        return population[0]

    def trace_path(self, individual):
        """Traces and visualizes the path taken by the solution."""
        curr = list(self.start)
        path = [tuple(curr)]
        
        for move_idx in individual:
            dr, dc = self.directions[move_idx]
            nr, nc = curr[0] + dr, curr[1] + dc;
            
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                curr = [nr, nc]
                path.append(tuple(curr))
                
                if tuple(curr) == self.goal:
                    break;
        
        return path

# --- Main Execution ---

grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

print("--- Map Layout ---")
print("(0 = walkable, 1 = wall)")
for i, r in enumerate(grid):
    print(f"Row {i}: {r}")

try:
    print("\nEnter coordinates (e.g., 0,0)")
    start_in = input("Start Point (row,col): ").strip().split(',')
    goal_in = input("Goal Point (row,col): ").strip().split(',')
    
    start_node = (int(start_in[0].strip()), int(start_in[1].strip()))
    goal_node = (int(goal_in[0].strip()), int(goal_in[1].strip()))

    # Validate coordinates are within bounds
    if not (0 <= start_node[0] < len(grid) and 0 <= start_node[1] < len(grid[0])):
        print("Error: Start coordinates out of bounds.")
    elif not (0 <= goal_node[0] < len(grid) and 0 <= goal_node[1] < len(grid[0])):
        print("Error: Goal coordinates out of bounds.")
    elif grid[start_node[0]][start_node[1]] == 1:
        print("Error: Start is on a wall.")
    elif grid[goal_node[0]][goal_node[1]] == 1:
        print("Error: Goal is on a wall.")
    else:
        print("\nEvolving path... please wait.")
        ga = GeneticPathfinder(grid, start_node, goal_node)
        best_path_genes = ga.run()
        print("\nEvolution complete!")
        
        # Display results
        path = ga.trace_path(best_path_genes)
        fitness = ga.calculate_fitness(best_path_genes)
        print(f"Best Fitness: {fitness:.4f}")
        print(f"Path Length: {len(path)} steps")
        print(f"Path: {path}")
        print(f"Best Move Sequence (0=U, 1=D, 2=L, 3=R): {best_path_genes[:20]}...")

except (ValueError, IndexError) as e:
    print(f"Invalid format. Use row,col (e.g., 0,0). Error: {e}")
