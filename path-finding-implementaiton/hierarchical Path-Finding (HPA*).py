import heapq

class HPAStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, a, b):
        # Manhattan distance for grid movement
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        r, c = node
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            # Check bounds and if the tile is walkable (0)
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == 0:
                yield (nr, nc), 1

    def solve(self, start, goal):
        # Priority Queue for A* search: (priority, current_node)
        pq = [(0, start)]
        g_score = {start: 0}
        parent = {start: None}
        
        while pq:
            _, current = heapq.heappop(pq)
            
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1], g_score[goal]
            
            for neighbor, cost in self.get_neighbors(current):
                new_g = g_score[current] + cost
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    priority = new_g + self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (priority, neighbor))
                    parent[neighbor] = current
                    
        return None, float('inf')

# --- Main Execution ---

# 0 = Path, 1 = Wall
grid = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

print("--- Current Grid Map ---")
for row in grid:
    print(row)

try:
    # User Input Handling
    print("\nEnter coordinates in 'row,col' format (e.g., 0,0)")
    
    start_input = input("Enter Start Point: ").split(',')
    goal_input = input("Enter Goal Point: ").split(',')
    
    start = (int(start_input[0]), int(start_input[1]))
    goal = (int(goal_input[0]), int(goal_input[1]))

    # Validation
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print("Error: Start or Goal is inside a wall (1).")
    else:
        solver = HPAStar(grid)
        path, cost = solver.solve(start, goal)
        
        if path:
            print(f"\nPath successfully found! Total Cost: {cost}")
            print(f"Path sequence: {path}")
        else:
            print("\nNo valid path exists between these points.")

except (ValueError, IndexError):
    print("Invalid Input Format. Please use 'row,col' (e.g., 0,5)")
