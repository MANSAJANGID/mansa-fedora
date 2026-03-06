# ==========================================
# STEP 1: INSTALLATION
# ==========================================
!pip install -U langgraph langchain_core

import os
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END

# ==========================================
# STEP 2: THE DIGITAL MAP (The Graph)
# ==========================================
# A dictionary representing nodes and their connections. 
# Every node should ideally have a return path to prevent getting stuck.
DIGITAL_MAP = {
    "Home": ["Profile", "Settings", "Dashboard"],
    "Dashboard": ["Analytics", "Revenue", "Home"],
    "Settings": ["Security", "Billing", "Home"],
    "Security": ["Password_Reset", "Two_Factor", "Settings"],
    "Billing": ["Invoices", "Subscription_Plan", "Settings"],
    "Invoices": ["PDF_Generator", "Email_Service", "Billing"],
    "Profile": ["Home"],
    "Analytics": ["Dashboard"]
}

# ==========================================
# STEP 3: THE AGENT'S LOGIC
# ==========================================
class AgentState(TypedDict):
    current_node: str
    goal_node: str
    path: List[str]

def navigator(state: AgentState):
    current = state['current_node']
    goal = state['goal_node']
    
    # Use .get() to safely retrieve neighbors
    neighbors = DIGITAL_MAP.get(current, [])
    
    print(f"🤖 Agent at: [{current}] | Looking for: [{goal}]")
    
    # SAFETY GATE: Prevents IndexError if a node has no neighbors
    if not neighbors:
        print(f"❌ DEAD END at {current}!")
        return {"current_node": current, "path": state['path'] + ["STUCK"]}

    # Decide the next move
    if goal in neighbors:
        next_node = goal
    else:
        # Avoid circular loops by picking unvisited nodes
        unvisited = [n for n in neighbors if n not in state['path']]
        # If unvisited exists, take the first one; else backtrack
        next_node = unvisited[0] if unvisited else neighbors[0]

    return {
        "current_node": next_node,
        "path": state['path'] + [next_node]
    }

# Logic to decide if the agent should keep moving or stop
def should_continue(state: AgentState):
    if state['current_node'] == state['goal_node'] or "STUCK" in state['path']:
        return "end"
    if len(state['path']) > 15: # Prevent infinite loops
        return "end"
    return "continue"

# ==========================================
# STEP 4: BUILD & EXECUTE
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("move", navigator)
builder.set_entry_point("move")

builder.add_conditional_edges(
    "move", 
    should_continue, 
    {"continue": "move", "end": END}
)

# Compile the workflow into a runnable agent
agent_ai = builder.compile()

# Start the agent
initial_input = {
    "current_node": "Home", 
    "goal_node": "Invoices", 
    "path": ["Home"]
}

print("--- Starting Journey ---\n")
final_state = agent_ai.invoke(initial_input)

print("\n--- RESULTS ---")
print(f"Path Taken: {' ➔ '.join(final_state['path'])}")
