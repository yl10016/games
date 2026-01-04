"""
Greedy Driver Agent with A* Heuristic

This module implements a non-learning driver agent that uses A* heuristic
to make routing decisions based on beliefs about edge congestion.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class GreedyDriver:
    """
    A greedy driver that uses A* heuristic for next-move selection.
    
    The driver maintains beliefs about edge congestion and uses these beliefs
    to compute expected travel times. It then selects the next edge using
    A* heuristic (g(n) + h(n)) where:
    - g(n) = believed travel time on the edge
    - h(n) = Euclidean distance heuristic to destination
    """
    
    def __init__(self, driver_id: int, graph, congestion_params: Dict[str, float]):
        """
        Initialize a greedy driver.
        
        Args:
            driver_id: Unique identifier for this driver
            graph: RoadNetwork instance containing the graph structure
            congestion_params: Dictionary with 'alpha' and 'beta' for congestion function
        """
        self.driver_id = driver_id
        self.graph = graph
        self.alpha = congestion_params['alpha']
        self.beta = congestion_params['beta']
        
        # Driver state
        self.current_node = None
        self.destination = None
        self.beliefs = {}  # edge_id -> believed_congestion
        self.visited_nodes = []  # Track recently visited nodes to avoid loops
        self.max_history = 3  # Remember last 3 nodes
        self.last_edge = None  # Track the edge taken in the previous timestep
        
    def reset(self, start_node: int, destination: int):
        """Reset driver for a new trip."""
        self.current_node = start_node
        self.destination = destination
        self.beliefs = {}
        self.visited_nodes = [start_node]
        self.last_edge = None
        
    def set_beliefs(self, beliefs: Dict[int, float]):
        """
        Update driver's beliefs about edge congestion.
        
        Args:
            beliefs: Dictionary mapping edge_id -> believed_congestion
        """
        self.beliefs = beliefs.copy()
        
    def compute_travel_time(self, edge_id: int, congestion: float) -> float:
        """
        Compute travel time on an edge given congestion level.
        
        Uses the BPR (Bureau of Public Roads) function:
        τ_e(x_e) = τ_e^0 * (1 + α * (x_e)^β)
        
        Args:
            edge_id: Edge identifier
            congestion: Number of drivers on the edge
            
        Returns:
            Travel time on the edge
        """
        base_time = self.graph.get_base_time(edge_id)
        return base_time * (1 + self.alpha * (congestion ** self.beta))
        
    def euclidean_distance(self, node_a: int, node_b: int) -> float:
        """
        Compute Euclidean distance between two nodes.
        Scaled appropriately to be proportional to g.
        
        Args:
            node_a: First node ID
            node_b: Second node ID
            
        Returns:
            Euclidean distance
        """
        pos_a = self.graph.get_node_position(node_a)
        pos_b = self.graph.get_node_position(node_b)
        scale = 3.0
        return scale * np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
        
    def compute_astar_next_move(self) -> Optional[int]:
        """
        Compute the next edge to take using A* heuristic with loop avoidance.
        
        For each neighboring edge from current position:
        1. Compute g(n) = believed travel time on edge
        2. Compute h(n) = Euclidean distance from edge destination to goal
        3. Add penalty for recently visited nodes to avoid loops
        4. Select edge with minimum f(n) = g(n) + h(n) + loop_penalty
        
        Returns:
            Edge ID to take next, or None if no valid moves
        """
        if self.current_node == self.destination:
            return None
            
        # Get all outgoing edges from current node
        neighbor_edges = self.graph.get_outgoing_edges(self.current_node)
        
        if not neighbor_edges:
            return None
            
        best_edge = None
        best_score = float('inf')
        
        for edge_id in neighbor_edges:
            # Get destination node of this edge
            next_node = self.graph.get_edge_destination(edge_id, self.current_node)
            
            # Compute g(n): believed travel time on this edge
            believed_congestion = self.beliefs.get(edge_id, 0.0)
            g_score = self.compute_travel_time(edge_id, believed_congestion)
            
            # Compute h(n): Euclidean distance heuristic
            h_score = self.euclidean_distance(next_node, self.destination)
            
            # Add large penalty for recently visited nodes to avoid loops
            loop_penalty = 0.0
            if next_node in self.visited_nodes[-self.max_history:]:
                loop_penalty = 1000.0
            
            # Compute f(n) = g(n) + h(n) + loop_penalty
            f_score = g_score + h_score + loop_penalty
            
            if f_score < best_score:
                best_score = f_score
                best_edge = edge_id
                
        return best_edge
        
    def get_current_node(self) -> int:
        """Get current node position."""
        return self.current_node
    
    def update_position(self, new_node: int, edge_taken: Optional[int] = None):
        """Update driver's current position and track visited nodes."""
        self.current_node = new_node
        self.last_edge = edge_taken
        self.visited_nodes.append(new_node)
        # Keep only recent history to avoid memory growth
        if len(self.visited_nodes) > 10:
            self.visited_nodes = self.visited_nodes[-10:]
    
    def get_destination(self) -> int:
        """Get destination node."""
        return self.destination
        
    def get_last_edge(self) -> Optional[int]:
        """Get the edge taken in the previous timestep."""
        return self.last_edge
        
    def has_reached_destination(self) -> bool:
        """Check if driver has reached destination."""
        return self.current_node == self.destination
