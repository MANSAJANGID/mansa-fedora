from collections import deque

def graph_fringe_search(start, goal, graph, heuristics):
    """
    Graph-based Fringe Search.
    graph: Adjacency list {node: {neighbor: cost}}
    heuristics: Dictionary of estimated costs to the goal {node: h_cost}
    """
    # Helper to safely get heuristic (defaults to 0 if not provided, acting like Dijkstra)
    def get_h(node):
        return heuristics.get(node, 0)

    # now stores: (f_cost, g_cost, current_node)
    now = deque([(get_h(start), 0, start)])
    later = deque()
    
    threshold = get_h(start)
    visited = {start: (0, None)} # {node: (best_g_cost, parent_node)}

    while now or later:
        min_f_exceeded = float('inf')
        
        while now:
            f, g, current = now.popleft()

            # If the current f_cost exceeds our threshold, defer it to the next pass
            if f > threshold:
                min_f_exceeded = min(min_f_exceeded, f)
                later.append((f, g, current))
                continue

            # Goal reached! Backtrack to build the path.
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current][1]
                return path[::-1], g

            # Expand neighbors using the Graph's adjacency list
            for neighbor, step_cost in graph.get(current, {}).items():
                new_g = g + step_cost
                new_f = new_g + get_h(neighbor)
                
                # If we found a cheaper way to this neighbor, update and explore
                if neighbor not in visited or new_g < visited[neighbor][0]:
                    visited[neighbor] = (new_g, current)
                    # Use appendleft to maintain the DFS-like expansion of Fringe Search
                    now.appendleft((new_f, new_g, neighbor))

        # If we exhausted the 'now' list and 'later' is empty, no path exists
        if not later:
            break
            
        # Update threshold for the next iteration and swap lists
        threshold = min_f_exceeded
        now = later
        later = deque()

    return None, float('inf')


def main():
    # --- Define the Graph Environment ---
    # Adjacency List: {Node: {Neighbor: Edge_Cost}}
    graph = {
        'A': {'B': 2, 'C': 5, 'D': 10},
        'B': {'A': 2, 'C': 2, 'E': 4},
        'C': {'A': 5, 'B': 2, 'D': 1, 'GOAL': 8},
        'D': {'A': 10, 'C': 1, 'F': 2},
        'E': {'B': 4, 'GOAL': 3},
        'F': {'D': 2, 'GOAL': 6},
        'GOAL': {}
    }

    # Pre-calculated heuristic estimates to the 'GOAL'
    # For a perfect search, these must never overestimate the true cost
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
    print("Fringe Search - Graph Pathfinding")
    print("=" * 50)
    print("\nGraph Connections & Edge Costs:")
    for node, edges in graph.items():
        print(f"Node {node} -> {edges}")

    valid_nodes = list(graph.keys())

    try:
        print(f"\nValid Nodes: {valid_nodes}")
        
        # User Input for Start Node
        while True:
            start_node = input("\nEnter Start Node: ").strip().upper()
            if start_node in valid_nodes:
                break
            print(f"Invalid. Please choose from {valid_nodes}")
        
        # User Input for Goal Node
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
        print("\nSearching for optimal path...\n")

        # Execute Graph Fringe Search
        final_path, total_cost = graph_fringe_search(start_node, goal_node, graph, heuristics)

        print("=" * 50)
        if final_path:
            print(f"SUCCESS: Optimal Path Found!")
            print(f"Total Cost: {total_cost}")
            print(f"Path: {' -> '.join(final_path)}")
        else:
            print("FAILED: No path found.")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
