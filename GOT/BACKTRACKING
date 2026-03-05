import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END

# --- 1. THE DIGITAL MAP (GoT Structure) ---
# We define a graph where some paths are dead ends (like 'Profile') 
# [span_3](start_span)[span_4](start_span)to force the agent to refine its path[span_3](end_span)[span_4](end_span).
DIGITAL_MAP = {
    "Home": ["Profile", "Settings", "Dashboard"],
    "Dashboard": ["Analytics", "Revenue", "Home"],
    "Settings": ["Security", "Billing", "Home"],
    "Security": ["Password_Reset", "Two_Factor", "Settings"],
    "Billing": ["Invoices", "Settings"],
    "Invoices": ["Goal_Node"],
    "Profile": [],           # Dead End 1
    "Password_Reset": [],    # Dead End 2
    "Goal_Node": []          # Target
}

# --- 2. THE GRAPH REASONING STATE (GRS) ---
# [span_5](start_span)[span_6](start_span)In GoT, the state must maintain a history to enable backtracking[span_5](end_span)[span_6](end_span).
class AgentState(TypedDict):
    current_node: str
    goal_node: str
    path_stack: List[str]  # The 'Stack' for backtracking
    visited: List[str]     # Global history to avoid re-entering dead ends

# --- 3. THE NAVIGATOR (Refining Transformation) ---
def navigator(state: AgentState):
    current = state['current_node']
    goal = state['goal_node']
    neighbors = DIGITAL_MAP.get(current, [])
    
    print(f"🤖 Agent at: [{current}]")

    # [span_7](start_span)If we reached the goal, stop transformations[span_7](end_span)
    if current == goal:
        return state

    # [span_8](start_span)[span_9](start_span)Find neighbors not yet in our global 'visited' history[span_8](end_span)[span_9](end_span)
    options = [n for n in neighbors if n not in state['visited']]

    if options:
        # [span_10](start_span)GENERATION: Move forward to the next thought[span_10](end_span)
        next_node = options[0]
        return {
            "current_node": next_node,
            "path_stack": state['path_stack'] + [current],
            "visited": state['visited'] + [next_node]
        }
    else:
        # [span_11](start_span)REFINING: Backtrack to the previous valid thought[span_11](end_span)
        if not state['path_stack']:
            return {"current_node": "FAILED"}
        
        prev_node = state['path_stack'][-1]
        print(f"🔄 Dead end! Refining path by backtracking to {prev_node}...")
        return {
            "current_node": prev_node,
            "path_stack": state['path_stack'][:-1],
            "visited": state['visited']
        }

# --- 4. CONTROLLER (Graph Logic) ---
def should_continue(state: AgentState):
    if state['current_node'] == state['goal_node'] or state['current_node'] == "FAILED":
        return "end"
    return "continue"

# [span_12](start_span)[span_13](start_span)Build the GoT Workflow[span_12](end_span)[span_13](end_span)
builder = StateGraph(AgentState)
builder.add_node("move", navigator)
builder.set_entry_point("move")
builder.add_conditional_edges("move", should_continue, {"continue": "move", "end": END})

agent_app = builder.compile()

# --- 5. EXECUTION ---
inputs = {"current_node": "Home", "goal_node": "Goal_Node", "path_stack": [], "visited": ["Home"]}
result = agent_app.invoke(inputs)
print(f"\n✅ Path successfully found: {' -> '.join(result['visited'])}")
