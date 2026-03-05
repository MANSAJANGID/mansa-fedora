import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# --- 1. Random Graph Generator ---
def generate_random_graph(num_nodes=15, extra_edges=20):
    nodes = [str(i) for i in range(num_nodes)]
    graph = {node: {} for node in nodes}
    
    connected = [nodes[0]]
    unconnected = nodes[1:]
    random.shuffle(unconnected)
    
    for new_node in unconnected:
        target = random.choice(connected)
        weight = random.randint(1, 15)
        graph[target][new_node] = weight
        graph[new_node][target] = weight 
        connected.append(new_node)
        
    for _ in range(extra_edges):
        u, v = random.sample(nodes, 2)
        if v not in graph[u]:
            weight = random.randint(1, 15)
            graph[u][v] = weight
            graph[v][u] = weight
            
    start, goal = random.sample(nodes, 2)
    return graph, start, goal

# --- 2. The Pathfinding Algorithm (WITH HISTORY TRACKING) ---
def solve_graph_with_history(graph, start, goal):
    pq = [(0, start)]
    g_score = {start: 0}
    parent = {start: None}
    
    history = [] # This will store the "frames" of our animation
    visited_set = set()
    
    while pq:
        current_cost, current_node = heapq.heappop(pq)
        
        # Skip if we've already finalized this node
        if current_node in visited_set:
            continue
            
        visited_set.add(current_node)
        
        # Figure out the path taken to get here so far
        curr = current_node
        partial_path = []
        while curr is not None:
            partial_path.append(curr)
            curr = parent.get(curr)
        partial_path = partial_path[::-1]
        
        # Get nodes currently waiting in the priority queue
        frontier_nodes = [n for c, n in pq]
        
        # RECORD THE FRAME
        history.append({
            'current': current_node,
            'visited': list(visited_set),
            'frontier': frontier_nodes,
            'path': partial_path,
            'is_goal': current_node == goal
        })
        
        if current_node == goal:
            return partial_path, g_score[goal], history
            
        for neighbor, weight in graph[current_node].items():
            if neighbor in visited_set:
                continue
            new_cost = g_score[current_node] + weight
            if neighbor not in g_score or new_cost < g_score[neighbor]:
                g_score[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))
                parent[neighbor] = current_node
                
    return None, float('inf'), history

# --- 3. The Interactive Visualizer ---
def animate_search(graph_data, history, start, goal):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    G = nx.Graph()
    for u, neighbors in graph_data.items():
        for v, weight in neighbors.items():
            G.add_edge(u, v, weight=weight)
            
    # FIXED SEED: This is crucial so the nodes don't jump around in every frame
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) 
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # This function is called for every frame in the history
    def update(frame_index):
        ax.clear()
        state = history[frame_index]
        
        # Base Graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=800)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        # Read the state data
        visited = state['visited']
        frontier = state['frontier']
        current = state['current']
        path = state['path']
        is_goal = state['is_goal']
        
        # Draw Visited Nodes (Light Blue)
        if visited:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=visited, node_color='lightblue', node_size=800)
            
        # Draw Frontier/Queue Nodes (Yellow)
        if frontier:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=frontier, node_color='yellow', node_size=800)
            
        # Keep Start Green and Goal Salmon
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[start], node_color='lightgreen', node_size=1000)
        if not is_goal:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[goal], node_color='salmon', node_size=1000)
            
        # Highlight Current Node (Magenta)
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[current], node_color='magenta', node_size=1200)
        
        # Draw the active path tracking
        if len(path) > 1:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges, edge_color='blue', width=4)
            
        if is_goal:
            ax.set_title(f"Step {frame_index+1}/{len(history)}: Goal Reached!", color='green', fontsize=16, fontweight='bold')
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=path, node_color='gold', node_size=1200)
        else:
            ax.set_title(f"Step {frame_index+1}/{len(history)}: Evaluating Node '{current}'\nYellow = In Queue | Blue = Visited", fontsize=14)

    # Generate the animation
    print("Generating interactive player... this might take a few seconds.")
    anim = FuncAnimation(fig, update, frames=len(history), interval=600, repeat=False)
    
    # Close the static plot so we only see the interactive widget
    plt.close()
    
    # Render the JS HTML widget
    return HTML(anim.to_jshtml())

# --- 4. Main Execution ---
random_graph, random_start, random_goal = generate_random_graph(num_nodes=15, extra_edges=20)

print(f"Start Node: {random_start}")
print(f"Goal Node: {random_goal}")

final_path, total_cost, search_history = solve_graph_with_history(random_graph, random_start, random_goal)

if final_path:
    print(f"Path found! Cost: {total_cost}")
else:
    print("No path found.")

# Display the interactive player
display(animate_search(random_graph, search_history, random_start, random_goal))
