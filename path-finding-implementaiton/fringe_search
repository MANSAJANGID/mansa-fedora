from collections import deque

def fringe_search(start, goal, heuristic, neighbors_func):
    """
    Highly efficient Fringe Search implementation.
    Uses parent-pointers to avoid O(n^2) path copying.
    """
    # Each entry in Now/Later is (f_cost, g_cost, node)
    now = deque([(heuristic(start, goal), 0, start)])
    later = deque()
    
    threshold = heuristic(start, goal)
    
    # Stores {node: (best_g_cost, parent_node)}
    visited = {start: (0, None)}

    while now or later:
        min_f_exceeded = float('inf')
        
        while now:
            f, g, current = now.popleft()

            if f > threshold:
                min_f_exceeded = min(min_f_exceeded, f)
                later.append((f, g, current))
                continue

            if current == goal:
                # Build the final path by backtracking parents
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current][1]
                return path[::-1], g

            for neighbor, step_cost in neighbors_func(current):
                new_g = g + step_cost
                new_f = new_g + heuristic(neighbor, goal)
                
                if neighbor not in visited or new_g < visited[neighbor][0]:
                    visited[neighbor] = (new_g, current)
                    now.append((new_f, new_g, neighbor))

        if not later:
            break;
            
        threshold = min_f_exceeded
        now = later
        later = deque()

    return None, float('inf')

# --- Helper Functions ---

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_grid_neighbors(node, grid):
    rows, cols = len(grid), len(grid[0])
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            yield (nx, ny), 1

# --- Execution ---
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

start_pos = (0, 0)
goal_pos = (4, 4)

# We wrap the neighbors function to pass the grid context
path, cost = fringe_search(
    start_pos, 
    goal_pos, 
    manhattan, 
    lambda n: get_grid_neighbors(n, grid)
)

print(f"Path: {path}")
print(f"Total Cost: {cost}")
