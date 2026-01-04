"""
Gym-style environment for traffic routing with information design
- Operator observes full state and provides regional traffic information
- Drivers make routing decisions based on beliefs updated by operator signals
- Reward is based on system-wide traffic efficiency (total delay)
"""

import json
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from driver import GreedyDriver


# Regional traffic signal action space
ACTION_STRATEGIES = [
    "no_information",
    "northwest_congested",
    "northwest_light",
    "northeast_congested",
    "northeast_light",
    "southwest_congested",
    "southwest_light",
    "southeast_congested",
    "southeast_light",
    "center_congested",
    "center_light",
    "north_south_corridor_congested",
    "north_south_corridor_light",
    "east_west_corridor_congested",
    "east_west_corridor_light",
]


class RoadNetwork:
    """
    Road network represented as a directed graph.
    
    Manages the graph structure, spatial coordinates, edge properties,
    and regional groupings for information signals.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize road network from configuration.
        
        Args:
            config: Dictionary containing:
                - nodes: List of [node_id, x, y]
                - edges: List of [source, target, base_time]
                - regions: Dictionary mapping region names to node lists
                - congestion_params: alpha and beta parameters
        """
        self.G = nx.DiGraph()
        self.edge_id_to_tuple = {}
        self.edge_tuple_to_id = {}
        self.node_positions = {}
        self.base_times = {}
        self.regions = config.get('regions', {})
        
        self._build_graph(config)
        
    def _build_graph(self, config: Dict):
        """Build the graph from configuration."""
        # Add nodes with positions
        for node_id, x, y in config['nodes']:
            self.G.add_node(node_id, pos=(x, y))
            self.node_positions[node_id] = (x, y)
        
        # Add edges with base travel times
        # Handle case where edges need to be auto-generated (8x8 grid)
        edges = config.get('edges', [])
        if not edges:
            edges = self._generate_grid_edges(config['nodes'])
            
        for edge_id, (source, target, base_time) in enumerate(edges):
            self.G.add_edge(source, target, edge_id=edge_id, base_time=base_time)
            self.edge_id_to_tuple[edge_id] = (source, target)
            self.edge_tuple_to_id[(source, target)] = edge_id
            self.base_times[edge_id] = base_time
            
    def _generate_grid_edges(self, nodes: List) -> List:
        """Auto-generate edges for a grid network."""
        edges = []
        node_ids = [n[0] for n in nodes]
        positions = {n[0]: (n[1], n[2]) for n in nodes}
        
        # Create edges between nodes that are distance 1.0 apart
        for node_a in node_ids:
            for node_b in node_ids:
                if node_a >= node_b:
                    continue
                pos_a = positions[node_a]
                pos_b = positions[node_b]
                dist = np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
                if abs(dist - 1.0) < 0.01:  # Neighboring nodes
                    edges.append([node_a, node_b, 5.0])
                    edges.append([node_b, node_a, 5.0])
        return edges
        
    def get_num_nodes(self) -> int:
        """Get number of nodes."""
        return self.G.number_of_nodes()
        
    def get_num_edges(self) -> int:
        """Get number of edges."""
        return len(self.edge_id_to_tuple)
        
    def get_node_position(self, node_id: int) -> Tuple[float, float]:
        """Get (x, y) position of a node."""
        return self.node_positions[node_id]
        
    def get_base_time(self, edge_id: int) -> float:
        """Get base travel time for an edge."""
        return self.base_times[edge_id]
        
    def get_outgoing_edges(self, node_id: int) -> List[int]:
        """Get list of outgoing edge IDs from a node."""
        edge_ids = []
        for target in self.G.successors(node_id):
            edge_id = self.G[node_id][target]['edge_id']
            edge_ids.append(edge_id)
        return edge_ids
        
    def get_edge_destination(self, edge_id: int, source_node: int) -> int:
        """Get destination node of an edge given source node."""
        source, target = self.edge_id_to_tuple[edge_id]
        return target if source == source_node else source
        
    def get_region_nodes(self, region_name: str) -> List[int]:
        """Get list of nodes in a region."""
        return self.regions.get(region_name, [])
        
    def get_region_edges(self, region_name: str) -> List[int]:
        """Get list of edges within a region (both endpoints in region)."""
        region_nodes = set(self.get_region_nodes(region_name))
        region_edges = []
        for edge_id, (source, target) in self.edge_id_to_tuple.items():
            if source in region_nodes and target in region_nodes:
                region_edges.append(edge_id)
        return region_edges
        
    def get_all_nodes(self) -> List[int]:
        """Get list of all node IDs."""
        return list(self.G.nodes())


class TrafficEnv(gym.Env):
    """
    Gym environment for traffic routing with information design.
    
    The operator (RL agent) sends regional traffic signals to drivers,
    who update their beliefs and make routing decisions accordingly.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        network_config_path: str,
        num_drivers: int = 10,
        max_steps: int = 100,
        belief_multiplier_congested: float = 2.0,
        belief_multiplier_light: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize traffic environment.
        
        Args:
            network_config_path: Path to JSON file with network configuration
            num_drivers: Number of drivers in the system
            max_steps: Maximum steps per episode
            belief_multiplier_congested: Multiplier for "congested" signals
            belief_multiplier_light: Multiplier for "light" signals
            seed: Random seed
        """
        super().__init__()
        
        # Load network configuration
        with open(network_config_path, 'r') as f:
            config = json.load(f)
            
        self.network = RoadNetwork(config)
        self.num_drivers = num_drivers
        self.max_steps = max_steps
        self.congestion_params = config['congestion_params']
        self.belief_multiplier_congested = belief_multiplier_congested
        self.belief_multiplier_light = belief_multiplier_light
        
        # Action space: discrete choice of regional signals
        self.action_space = spaces.Discrete(len(ACTION_STRATEGIES))
        
        # Observation space: [edge_congestion, driver_positions, driver_destinations]
        num_edges = self.network.get_num_edges()
        num_nodes = self.network.get_num_nodes()
        obs_dim = num_edges + num_drivers * 2  # congestion + positions + destinations
        self.observation_space = spaces.Box(
            low=0, high=max(num_drivers, num_nodes),
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize drivers
        self.drivers = [
            GreedyDriver(i, self.network, self.congestion_params)
            for i in range(num_drivers)
        ]
        
        # Episode state
        self.current_step = 0
        self.edge_congestion = np.zeros(num_edges)
        self.last_action = 0
        
        if seed is not None:
            self.seed(seed)
            
    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment for a new episode.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
            
        self.current_step = 0
        self.last_action = 0
        
        # Reset edge congestion
        num_edges = self.network.get_num_edges()
        self.edge_congestion = np.zeros(num_edges)
        
        # Assign random start/destination to each driver
        all_nodes = self.network.get_all_nodes()
        for driver in self.drivers:
            start = np.random.choice(all_nodes)
            dest = np.random.choice(all_nodes)
            while dest == start:
                dest = np.random.choice(all_nodes)
            driver.reset(start, dest)
            
        # Initialize beliefs with uniform distribution
        self._update_driver_beliefs(0)  # "no_information" action
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def _get_default_beliefs(self) -> Dict[int, float]:
        """
        Get default belief: uniform distribution of drivers across edges.
        
        Returns:
            Dictionary mapping edge_id -> expected_congestion
        """
        num_edges = self.network.get_num_edges()
        expected_congestion = self.num_drivers / num_edges if num_edges > 0 else 0
        return {i: expected_congestion for i in range(num_edges)}
        
    def _get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions based on actual congestion.
        Operator can only signal truthfully:
        - 'congested': actual congestion >= 1.5x expected
        - 'light': actual congestion <= 0.75x expected
        - 'no_information': always valid
        
        Returns:
            List of valid action indices
        """
        valid_actions = [0]  # 'no_information' is always valid
        
        num_edges = self.network.get_num_edges()
        expected_congestion_per_edge = self.num_drivers / num_edges if num_edges > 0 else 0
        
        # Check each region signal
        for action_idx, signal in enumerate(ACTION_STRATEGIES):
            if signal == "no_information":
                continue
                
            # Parse signal
            region_name = None
            signal_type = None
            
            if signal.endswith("_congested"):
                region_name = signal.replace("_congested", "")
                signal_type = "congested"
            elif signal.endswith("_light"):
                region_name = signal.replace("_light", "")
                signal_type = "light"
            
            if region_name:
                # Get edges in the region
                region_edges = self.network.get_region_edges(region_name)
                
                if len(region_edges) == 0:
                    continue
                
                # Calculate actual vs expected congestion in region
                actual_total = sum(self.edge_congestion[eid] for eid in region_edges)
                expected_total = expected_congestion_per_edge * len(region_edges)
                
                # Check if signal is truthful
                if signal_type == "congested" and actual_total >= 1.5 * expected_total:
                    valid_actions.append(action_idx)
                elif signal_type == "light" and actual_total <= 0.75 * expected_total:
                    valid_actions.append(action_idx)
        
        return valid_actions
    
    def _update_driver_beliefs(self, action: int):
        """
        Update all drivers' beliefs based on the operator's signal.
        
        Args:
            action: Index into ACTION_STRATEGIES
        """
        signal = ACTION_STRATEGIES[action]
        
        # Start with default uniform beliefs
        beliefs = self._get_default_beliefs()
        
        if signal == "no_information":
            # No update, keep uniform beliefs
            pass
        else:
            # Parse signal to get region and traffic level
            region_name = None
            multiplier = 1.0
            
            # Extract region (remove "_congested" or "_light" suffix)
            if signal.endswith("_congested"):
                region_name = signal.replace("_congested", "")
                multiplier = self.belief_multiplier_congested
            elif signal.endswith("_light"):
                region_name = signal.replace("_light", "")
                multiplier = self.belief_multiplier_light
                
            if region_name:
                # Get edges in the signaled region
                affected_edges = self.network.get_region_edges(region_name)
                
                # Update beliefs for affected edges
                for edge_id in affected_edges:
                    beliefs[edge_id] *= multiplier
                    
        # Set beliefs for all drivers
        for driver in self.drivers:
            driver.set_beliefs(beliefs)
            
    def _update_true_congestion(self):
        """Update true edge congestion based on edges drivers actually took."""
        num_edges = self.network.get_num_edges()
        self.edge_congestion = np.zeros(num_edges)
        
        # Count drivers on the edge they took in the previous timestep
        for driver in self.drivers:
            if not driver.has_reached_destination():
                last_edge = driver.get_last_edge()
                if last_edge is not None:
                    # Add +1 to the edge the driver actually took
                    self.edge_congestion[last_edge] += 1.0
                        
    def _compute_travel_time(self, edge_id: int, congestion: float) -> float:
        """Compute travel time on an edge."""
        base_time = self.network.get_base_time(edge_id)
        alpha = self.congestion_params['alpha']
        beta = self.congestion_params['beta']
        return base_time * (1 + alpha * (congestion ** beta))
        
    def _compute_operator_reward(self) -> float:
        """
        Compute operator's reward based on total delay.
        
        Reward = -sum(x_e * (tau_e(x_e) - tau_e^0))
        
        Returns:
            Negative total delay (higher is better)
        """
        total_delay = 0.0
        for edge_id in range(self.network.get_num_edges()):
            congestion = self.edge_congestion[edge_id]
            if congestion > 0:
                travel_time = self._compute_travel_time(edge_id, congestion)
                base_time = self.network.get_base_time(edge_id)
                delay = travel_time - base_time
                total_delay += congestion * delay
                
        return -total_delay
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation for the operator.
        
        Returns:
            Flattened array: [edge_congestion, driver_positions, driver_destinations]
        """
        # Edge congestion
        obs = list(self.edge_congestion)
        
        # Driver positions
        for driver in self.drivers:
            obs.append(float(driver.get_current_node()))
            
        # Driver destinations
        for driver in self.drivers:
            obs.append(float(driver.get_destination()))
            
        return np.array(obs, dtype=np.float32)
        
    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        num_reached = sum(1 for d in self.drivers if d.has_reached_destination())
        avg_congestion = np.mean(self.edge_congestion) if len(self.edge_congestion) > 0 else 0
        
        return {
            'step': self.current_step,
            'drivers_reached': num_reached,
            'total_drivers': self.num_drivers,
            'avg_congestion': avg_congestion,
            'last_signal': ACTION_STRATEGIES[self.last_action],
        }
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index into ACTION_STRATEGIES
            
        Returns:
            observation: New state
            reward: Operator's reward
            terminated: Whether episode is done (all drivers reached destination)
            truncated: Whether episode exceeded max steps
            info: Additional information
        """
        self.current_step += 1
        
        # Filter to valid actions only (truthful signals)
        valid_actions = self._get_valid_actions()
        if action not in valid_actions:
            # If action is invalid, default to 'no_information'
            action = 0
        
        self.last_action = action
        
        # 1. Operator sends signal, drivers update beliefs
        self._update_driver_beliefs(action)
        
        # 2. Each driver computes next move and updates position
        for driver in self.drivers:
            if not driver.has_reached_destination():
                next_edge = driver.compute_astar_next_move()
                if next_edge is not None:
                    next_node = self.network.get_edge_destination(
                        next_edge, driver.get_current_node()
                    )
                    driver.update_position(next_node, edge_taken=next_edge)
                    
        # 3. Update true congestion based on new positions
        self._update_true_congestion()
        
        # 4. Compute operator's reward
        reward = self._compute_operator_reward()
        
        # 5. Check termination conditions
        all_reached = all(d.has_reached_destination() for d in self.drivers)
        terminated = all_reached
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def render(self):
        """Render current state (console output)."""
        print(f"\n--- Step {self.current_step} ---")
        print(f"Last signal: {ACTION_STRATEGIES[self.last_action]}")
        print(f"Drivers reached destination: {self._get_info()['drivers_reached']}/{self.num_drivers}")
        print(f"Average congestion: {self._get_info()['avg_congestion']:.2f}")
