"""
Focused visualization on the slowest driver showing:
- Congestion on each edge (number labels)
- Heuristic cost for each neighbor edge (before/after operator signal)
- Operator signal with highlighted region
- Other cars as small numbered dots
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_env import TrafficEnv, ACTION_STRATEGIES
from operator_agent import OperatorAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize slowest driver decision-making')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to agent checkpoint (optional, uses random if not provided)')
    parser.add_argument('--network_config', type=str,
                        default='test_cases/grid_4x4.json',
                        help='Path to network configuration file')
    parser.add_argument('--num_drivers', type=int, default=15,
                        help='Number of drivers')
    parser.add_argument('--max_steps', type=int, default=30,
                        help='Maximum steps to visualize')
    parser.add_argument('--output_dir', type=str, default='test_cases/visualization_frames',
                        help='Directory to save individual frames')
    parser.add_argument('--save_animation', type=str, default='test_cases/traffic_animation.gif',
                        help='Path to save animation GIF')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def compute_heuristic_cost(driver, edge_id, network, congestion_params):
    """Compute A* heuristic cost (g + h) for an edge."""
    # g(n): travel time based on congestion
    base_time = network.base_times[edge_id]
    congestion = driver.beliefs.get(edge_id, 0.0)
    alpha = congestion_params['alpha']
    beta = congestion_params['beta']
    g_cost = base_time * (1 + alpha * (congestion ** beta))
    
    # h(n): Euclidean distance heuristic
    edge_tuple = network.edge_id_to_tuple[edge_id]
    next_node = edge_tuple[1]
    dest = driver.destination
    h_cost = driver.euclidean_distance(next_node, dest)
    
    return g_cost + h_cost


def collect_trajectory_with_details(env, agent, max_steps):
    """Collect detailed trajectory including before/after operator signal info."""
    state, _ = env.reset()
    trajectory_data = []
    
    for step in range(max_steps):
        # Store driver positions BEFORE this step
        positions_before = [d.get_current_node() for d in env.drivers]
        
        # Store state BEFORE operator signal (naive/default beliefs - uniform distribution)
        beliefs_before = {}
        default_beliefs = env._get_default_beliefs()
        for driver in env.drivers:
            beliefs_before[driver.driver_id] = default_beliefs.copy()
        
        # Get valid actions
        valid_actions = env._get_valid_actions()
        
        # Get action
        if agent is not None:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = agent.get_action(state_tensor, deterministic=True, valid_actions=valid_actions)
        else:
            action = np.random.choice(valid_actions)
        
        # Update beliefs with operator's signal to capture the "after" state
        env._update_driver_beliefs(action)
        
        # Store state AFTER operator signal (informed beliefs with signal applied)
        beliefs_after = {}
        for driver in env.drivers:
            beliefs_after[driver.driver_id] = driver.beliefs.copy()
        
        # Complete the step (drivers move, congestion updates, etc.)
        # Manually do what step() does since we already called _update_driver_beliefs
        env.current_step += 1
        env.last_action = action
        
        # Drivers compute next move and update position
        for driver in env.drivers:
            if not driver.has_reached_destination():
                next_edge = driver.compute_astar_next_move()
                if next_edge is not None:
                    next_node = env.network.get_edge_destination(
                        next_edge, driver.get_current_node()
                    )
                    driver.update_position(next_node, edge_taken=next_edge)
        
        # Update true congestion based on new positions
        env._update_true_congestion()
        
        # Compute reward
        reward = env._compute_operator_reward()
        
        # Get next state
        next_state = env._get_observation()
        
        # Check termination
        terminated = all(d.has_reached_destination() for d in env.drivers)
        truncated = env.current_step >= env.max_steps
        
        # Store complete state including last edges taken
        trajectory_data.append({
            'step': step,
            'action': action,
            'signal': ACTION_STRATEGIES[action],
            'beliefs_before': beliefs_before,
            'beliefs_after': beliefs_after,
            'congestion': env.edge_congestion.copy(),
            'driver_positions': [d.get_current_node() for d in env.drivers],
            'driver_destinations': [d.get_destination() for d in env.drivers],
            'driver_last_edges': [d.get_last_edge() for d in env.drivers],
            'driver_reached': [d.has_reached_destination() for d in env.drivers],
            'reward': reward,
        })
        
        state = next_state
        
        if terminated or truncated:
            break
    
    return trajectory_data


def find_slowest_driver(trajectory_data, num_drivers):
    """Find the driver that takes longest to reach destination."""
    arrival_times = {}
    
    for step_data in trajectory_data:
        for driver_id in range(num_drivers):
            pos = step_data['driver_positions'][driver_id]
            dest = step_data['driver_destinations'][driver_id]
            
            if pos == dest and driver_id not in arrival_times:
                arrival_times[driver_id] = step_data['step']
    
    # Find driver with latest arrival (or never arrived)
    slowest_driver = None
    slowest_time = -1
    
    for driver_id in range(num_drivers):
        arrival_time = arrival_times.get(driver_id, float('inf'))
        if arrival_time > slowest_time:
            slowest_time = arrival_time
            slowest_driver = driver_id
    
    return slowest_driver


def get_region_nodes(network, region_name):
    """Get nodes in a region based on the signal name."""
    if region_name == 'no_information':
        return []
    
    # Direct region name lookup (no composite regions needed anymore)
    return network.regions.get(region_name, [])


def create_traffic_animation(env, agent, max_steps=30, output_dir='test_cases/visualization_frames'):
    """Create focused animation on slowest driver."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    trajectory_data = collect_trajectory_with_details(env, agent, max_steps)
    
    slowest_driver_id = find_slowest_driver(trajectory_data, env.num_drivers)
    print(f"Slowest driver: {slowest_driver_id}")
    
    # Get graph components
    G = env.network.G
    pos = env.network.node_positions
    congestion_params = env.congestion_params
    
    print("Creating animation frames...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    def create_frame(frame_idx):
        """Create a single frame focused on slowest driver."""
        ax.clear()
        ax.invert_yaxis()  # Invert y-axis so north is up and south is down
        
        data = trajectory_data[frame_idx]
        
        # Get slowest driver info
        slowest_pos = data['driver_positions'][slowest_driver_id]
        slowest_dest = data['driver_destinations'][slowest_driver_id]
        
        # Parse operator signal
        signal = data['signal']
        region_name = signal.replace('_congested', '').replace('_light', '')
        highlighted_nodes = get_region_nodes(env.network, region_name)
        
        # === Draw edges with congestion numbers ===
        for u, v in G.edges():
            edge_id = G[u][v]['edge_id']
            congestion = data['congestion'][edge_id]
            
            # Draw arrow for the edge (no line, just arrow)
            x_coords = [pos[u][0], pos[v][0]]
            y_coords = [pos[u][1], pos[v][1]]
            
            # Calculate direction and offset for bidirectional edges
            dx = x_coords[1] - x_coords[0]
            dy = y_coords[1] - y_coords[0]
            edge_len = np.sqrt(dx*dx + dy*dy)
            if edge_len > 0:
                # Perpendicular offset to avoid overlap on bidirectional edges
                perp_x = -dy / edge_len * 0.05
                perp_y = dx / edge_len * 0.05
            else:
                perp_x, perp_y = 0, 0
            
            # Start and end points for arrow with offset
            # Make arrows shorter by starting and ending closer to nodes
            arrow_start_ratio = 0.25  # Start 25% along the edge
            arrow_end_ratio = 0.75    # End 75% along the edge
            
            start_x = x_coords[0] + (x_coords[1] - x_coords[0]) * arrow_start_ratio + perp_x
            start_y = y_coords[0] + (y_coords[1] - y_coords[0]) * arrow_start_ratio + perp_y
            end_x = x_coords[0] + (x_coords[1] - x_coords[0]) * arrow_end_ratio + perp_x
            end_y = y_coords[0] + (y_coords[1] - y_coords[0]) * arrow_end_ratio + perp_y
            
            # Draw arrow (shorter arrow in middle of edge)
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.4),
                       zorder=2)
            
            # Compute costs for this edge (naive vs informed)
            from driver import GreedyDriver
            temp_driver = GreedyDriver(slowest_driver_id, env.network, congestion_params)
            temp_driver.current_node = u  # Source of the edge
            temp_driver.destination = slowest_dest
            
            # Compute costs: naive (before signal), informed (after signal), and actual (true congestion)
            temp_driver.beliefs = data['beliefs_before'][slowest_driver_id]
            cost_naive = compute_heuristic_cost(temp_driver, edge_id, env.network, congestion_params)
            
            temp_driver.beliefs = data['beliefs_after'][slowest_driver_id]
            cost_informed = compute_heuristic_cost(temp_driver, edge_id, env.network, congestion_params)
            
            # Actual cost using true congestion from environment
            actual_congestion = data['congestion'][edge_id]
            temp_driver.beliefs = {edge_id: actual_congestion}
            cost_actual = compute_heuristic_cost(temp_driver, edge_id, env.network, congestion_params)
            
            # Calculate position for cost label (closer to arrow, aligned with arrow direction)
            edge_mid_x = (x_coords[0] + x_coords[1]) / 2
            edge_mid_y = (y_coords[0] + y_coords[1]) / 2
            
            # Calculate perpendicular offset to place label just above arrow (much smaller offset)
            if edge_len > 0:
                # Perpendicular vector (rotated 90 degrees) - small offset to stay close to arrow
                perp_x = -dy / edge_len * 0.15
                perp_y = dx / edge_len * 0.15
                
                # Calculate rotation angle to align text with arrow
                angle_deg = np.arctan2(dy, dx) * 180 / np.pi
                # Keep text right-side up: if angle is more than 90 or less than -90, flip it
                if angle_deg > 90 or angle_deg < -90:
                    angle_deg += 180
            else:
                perp_x, perp_y = 0, 0.15
                angle_deg = 0
            
            label_x = edge_mid_x + perp_x
            label_y = edge_mid_y + perp_y
            
            # Display costs along the edge (N=naive, I=informed, A=actual)
            cost_text = f'N:{cost_naive:.2f} I:{cost_informed:.2f} A:{cost_actual:.2f}'
            ax.text(label_x, label_y, cost_text,
                   fontsize=8, ha='center', va='center', fontweight='bold',
                   rotation=angle_deg, rotation_mode='anchor',
                   zorder=12)
            
            # Label congestion count on edge (if any)
            label_x = (x_coords[0] * 0.6 + x_coords[1] * 0.4)
            label_y = (y_coords[0] * 0.6 + y_coords[1] * 0.4)
            
            if congestion > 0:
                ax.text(label_x, label_y, f'{int(congestion)}',
                       fontsize=10, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='red', linewidth=1.5),
                       zorder=5)
        
        # === Draw nodes ===
        for node in G.nodes():
            x, y = pos[node]
            
            # Highlight region nodes if signal is active
            if node in highlighted_nodes:
                # Yellow highlight for signaled region
                circle = Circle((x, y), 0.28, color='yellow', ec='orange',
                              linewidth=2, zorder=8, alpha=0.7)
                ax.add_patch(circle)
            
            # Draw node
            if node == slowest_pos:
                # Slowest driver's current position (blue)
                circle = Circle((x, y), 0.22, color='dodgerblue', ec='darkblue',
                              linewidth=2.5, zorder=10)
            elif node == slowest_dest:
                # Slowest driver's destination (green)
                circle = Circle((x, y), 0.22, color='limegreen', ec='darkgreen',
                              linewidth=2.5, zorder=10)
            else:
                # Regular node
                circle = Circle((x, y), 0.22, color='lightgray', ec='black',
                              linewidth=2, zorder=9)
            ax.add_patch(circle)
            
            # Node label
            ax.text(x, y, str(node), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='black', zorder=11)
        
        # === Draw arrow for slowest driver's previous move ===
        if not data['driver_reached'][slowest_driver_id]:
            last_edge = data['driver_last_edges'][slowest_driver_id]
            if last_edge is not None:
                edge_tuple = env.network.edge_id_to_tuple[last_edge]
                prev_node = edge_tuple[0]
                prev_x, prev_y = pos[prev_node]
                curr_x, curr_y = pos[slowest_pos]
                
                # Calculate perpendicular offset for arrow
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                edge_len = np.sqrt(dx*dx + dy*dy)
                if edge_len > 0:
                    perp_x = -dy / edge_len * 0.08
                    perp_y = dx / edge_len * 0.08
                else:
                    perp_x, perp_y = 0, 0
                
                # Draw thick arrow from previous to current position with offset
                ax.annotate('', xy=(curr_x + perp_x, curr_y + perp_y), 
                           xytext=(prev_x + perp_x, prev_y + perp_y),
                           arrowprops=dict(arrowstyle='->', lw=3, 
                                         color='blue', alpha=0.7),
                           zorder=11)
        
        # === Draw other cars as small numbered dots ===
        for driver_id in range(env.num_drivers):
            if driver_id == slowest_driver_id:
                continue  # Skip slowest driver (already drawn)
            
            driver_pos = data['driver_positions'][driver_id]
            driver_reached = data['driver_reached'][driver_id]
            x, y = pos[driver_pos]
            
            # Offset for multiple drivers at same node
            offset_x = (driver_id % 4) * 0.15 - 0.225
            offset_y = ((driver_id // 4) % 4) * 0.15 - 0.225
            x += offset_x
            y += offset_y
            
            # Small dot with number - green if reached, gray if active
            dot_color = 'green' if driver_reached else 'gray'
            circle = Circle((x, y), 0.08, color=dot_color, ec='white',
                          linewidth=1, zorder=7)
            ax.add_patch(circle)
            ax.text(x, y, str(driver_id), ha='center', va='center',
                   fontsize=6, fontweight='bold', color='white', zorder=8)
            
            # Draw arrow showing previous edge taken
            if not driver_reached:
                last_edge = data['driver_last_edges'][driver_id]
                if last_edge is not None:
                    edge_tuple = env.network.edge_id_to_tuple[last_edge]
                    prev_node = edge_tuple[0]
                    prev_x, prev_y = pos[prev_node]
                    
                    # Calculate perpendicular offset for arrow
                    dx = x - prev_x
                    dy = y - prev_y
                    edge_len = np.sqrt(dx*dx + dy*dy)
                    if edge_len > 0:
                        perp_x = -dy / edge_len * 0.08
                        perp_y = dx / edge_len * 0.08
                    else:
                        perp_x, perp_y = 0, 0
                    
                    # Draw arrow from previous position to current with offset
                    ax.annotate('', xy=(x + perp_x, y + perp_y), 
                               xytext=(prev_x + perp_x, prev_y + perp_y),
                               arrowprops=dict(arrowstyle='->', lw=1.5, 
                                             color='purple', alpha=0.6),
                               zorder=6)
        
        # === Title with operator signal ===
        signal_text = data['signal'].replace('_', ' ').title()
        ax.set_title(f'Step {data["step"]} | Operator Signal: {signal_text}\n' +
                    f'Focused on Driver {slowest_driver_id} (Blue=Current, Green=Destination)',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = output_path / f'frame_{frame_idx:04d}.png'
        plt.savefig(frame_path, dpi=120, bbox_inches='tight')
        print(f"  Frame {frame_idx + 1}/{len(trajectory_data)}")
        
        return []
    
    # Generate all frames
    for i in range(len(trajectory_data)):
        create_frame(i)
    
    plt.close()
    
    return None


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize environment
    print(f"Initializing environment with {args.num_drivers} drivers...")
    env = TrafficEnv(
        network_config_path=args.network_config,
        num_drivers=args.num_drivers,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    # Load agent if checkpoint provided
    agent = None
    if args.checkpoint:
        print(f"Loading agent from {args.checkpoint}...")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = OperatorAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_episode_steps=args.max_steps,
        )
        agent.load(args.checkpoint)
        agent.eval()
    else:
        print("No checkpoint provided, using random policy")
    
    create_traffic_animation(
        env, agent,
        max_steps=args.max_steps,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

