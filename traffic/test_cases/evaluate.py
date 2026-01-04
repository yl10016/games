"""
Evaluate a trained operator agent and compare against baseline policies.
"""

import argparse
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_env import TrafficEnv, ACTION_STRATEGIES
from operator_agent import OperatorAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate traffic operator agent')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to agent checkpoint')
    parser.add_argument('--network_config', type=str,
                        default='test_cases/grid_4x4.json',
                        help='Path to network configuration file')
    parser.add_argument('--num_drivers', type=int, default=5,
                        help='Number of drivers')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension (must match checkpoint)')
    parser.add_argument('--belief_multiplier_congested', type=float, default=2.0,
                        help='Belief multiplier for congested signals')
    parser.add_argument('--belief_multiplier_light', type=float, default=0.5,
                        help='Belief multiplier for light signals')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


class BaselinePolicy:
    """Baseline policies for comparison."""
    
    def __init__(self, policy_type: str, action_dim: int):
        """
        Initialize baseline policy.
        
        Args:
            policy_type: One of ['no_info', 'random']
            action_dim: Number of actions
        """
        self.policy_type = policy_type
        self.action_dim = action_dim
        
        self.no_info_action = ACTION_STRATEGIES.index("no_information")
        
    def get_action(self, state):
        """Either return no_info action or random action."""
        if self.policy_type == 'random':
            return np.random.randint(self.action_dim)
        return self.no_info_action


def evaluate_policy(policy, env, num_episodes=100, policy_name="Policy"):
    """
    Evaluate a policy.
    
    Args:
        policy: Policy object with get_action() method
        env: Traffic environment
        num_episodes: Number of episodes to evaluate
        policy_name: Name for logging
        
    Returns:
        Dictionary of metrics
    """
    episode_rewards = []
    episode_delays = []
    episode_lengths = []
    completion_rates = []
    action_counts = {i: 0 for i in range(env.action_space.n)}
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        if isinstance(policy, OperatorAgent):
            state = torch.FloatTensor(state)
            
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            # Get action
            if isinstance(policy, OperatorAgent):
                action = policy.get_action(state, deterministic=True)
            else:
                action = policy.get_action(state)
                
            action_counts[action] += 1
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            if isinstance(policy, OperatorAgent):
                next_state = torch.FloatTensor(next_state)
                
            episode_reward += reward
            step += 1
            done = terminated or truncated
            state = next_state
            
        episode_rewards.append(episode_reward)
        episode_delays.append(-episode_reward)  # Reward is negative delay
        episode_lengths.append(step)
        completion_rates.append(info['drivers_reached'] / info['total_drivers'])
        
    # Calculate statistics
    results = {
        'policy_name': policy_name,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_delay': np.mean(episode_delays),
        'std_delay': np.std(episode_delays),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_completion_rate': np.mean(completion_rates),
        'std_completion_rate': np.std(completion_rates),
        'action_distribution': action_counts,
    }
    
    return results


def print_results(results):
    """Pretty print evaluation results."""
    print(f"\n{'='*70}")
    print(f"Policy: {results['policy_name']}")
    print(f"{'='*70}")
    print(f"Reward:          {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}")
    print(f"Total Delay:     {results['mean_delay']:8.2f} ± {results['std_delay']:.2f}")
    print(f"Episode Length:  {results['mean_length']:8.1f} ± {results['std_length']:.1f}")
    print(f"Completion Rate: {results['mean_completion_rate']:7.1%} ± {results['std_completion_rate']:.1%}")
    
    print(f"\nAction Distribution:")
    total_actions = sum(results['action_distribution'].values())
    for action_idx, count in sorted(results['action_distribution'].items(), 
                                    key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / total_actions * 100
            print(f"  {ACTION_STRATEGIES[action_idx]:30s}: {pct:5.1f}%  ({count} times)")
    print(f"{'='*70}\n")


def compare_policies(all_results):
    """Print comparison table of all policies."""
    print(f"\n{'='*70}")
    print(f"POLICY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Policy':<25} {'Delay':>12} {'Completion':>12} {'Length':>12}")
    print(f"{'-'*70}")
    
    for results in sorted(all_results, key=lambda x: x['mean_delay']):
        print(f"{results['policy_name']:<25} "
              f"{results['mean_delay']:12.2f} "
              f"{results['mean_completion_rate']:11.1%} "
              f"{results['mean_length']:12.1f}")
    
    print(f"{'='*70}")
    
    # Calculate improvement
    if len(all_results) > 1:
        best = min(all_results, key=lambda x: x['mean_delay'])
        baseline = [r for r in all_results if r['policy_name'] == 'no_info'][0]
        improvement = (baseline['mean_delay'] - best['mean_delay']) / baseline['mean_delay'] * 100
        print(f"\nBest policy ({best['policy_name']}) reduces delay by {improvement:.1f}% vs No Information")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize environment
    print(f"Initializing environment...")
    env = TrafficEnv(
        network_config_path=args.network_config,
        num_drivers=args.num_drivers,
        max_steps=args.max_steps,
        belief_multiplier_congested=args.belief_multiplier_congested,
        belief_multiplier_light=args.belief_multiplier_light,
        seed=args.seed
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load trained agent
    print(f"Loading trained agent from {args.checkpoint}...")
    agent = OperatorAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_episode_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
    )
    agent.load(args.checkpoint)
    agent.eval()
    
    # Evaluate trained agent
    print(f"Evaluating trained agent over {args.num_episodes} episodes...")
    trained_results = evaluate_policy(
        agent, env, args.num_episodes, policy_name="Trained Agent"
    )
    print_results(trained_results)
    
    # Evaluate baseline (no info)
    print(f"Evaluating baseline policies...")
    all_results = [trained_results]
    
    baselines = ['no_info', 'random']
    
    for policy_type in baselines:
        baseline = BaselinePolicy(policy_type, action_dim)
        results = evaluate_policy(baseline, env, args.num_episodes, policy_name=policy_type)
        print_results(results)
        all_results.append(results)
    
    compare_policies(all_results)
