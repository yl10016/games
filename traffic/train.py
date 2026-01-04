"""
Train the operator agent using Actor-Critic algorithm to learn
optimal information revelation strategies for traffic management.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from traffic_env import TrafficEnv
from operator_agent import OperatorAgent, collect_trajectory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train traffic operator agent')
    
    # Environment arguments
    parser.add_argument('--network_config', type=str, 
                        default='test_cases/grid_4x4.json',
                        help='Path to network configuration file')
    parser.add_argument('--num_drivers', type=int, default=15,
                        help='Number of drivers in the system')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--belief_multiplier_congested', type=float, default=2.0,
                        help='Belief multiplier for congested signals')
    parser.add_argument('--belief_multiplier_light', type=float, default=0.5,
                        help='Belief multiplier for light signals')
    
    # Training arguments
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--n_step', type=int, default=10,
                        help='N-step returns')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    
    # Logging arguments
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--save_freq', type=int, default=500,
                        help='Model save frequency (episodes)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def evaluate_agent(agent, env, num_eval_episodes=10):
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained operator agent
        env: Traffic environment
        num_eval_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    completion_rates = []
    
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        
        episode_reward = 0
        step = 0
        done = False
        
        while not done:
            # Use deterministic policy for evaluation
            action = agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state)
            
            episode_reward += reward
            step += 1
            done = terminated or truncated
            state = next_state
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        completion_rates.append(info['drivers_reached'] / info['total_drivers'])
        
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_completion_rate': np.mean(completion_rates),
    }


def train(args):
    """Main training loop."""
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize environment
    print(f"Initializing environment with {args.num_drivers} drivers...")
    env = TrafficEnv(
        network_config_path=args.network_config,
        num_drivers=args.num_drivers,
        max_steps=args.max_steps,
        belief_multiplier_congested=args.belief_multiplier_congested,
        belief_multiplier_light=args.belief_multiplier_light,
        seed=args.seed
    )
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Initializing agent (state_dim={state_dim}, action_dim={action_dim})...")
    agent = OperatorAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        lr=args.lr,
        max_episode_steps=args.max_steps,
        n_step=args.n_step,
        hidden_dim=args.hidden_dim,
    )
    
    # Training metrics
    episode_rewards = []
    policy_losses = []
    value_losses = []
    
    print(f"\nStarting training for {args.num_episodes} episodes...\n")
    
    # Training loop
    for episode in tqdm(range(args.num_episodes)):
        # Collect trajectory
        states, actions, rewards, next_states, terminated = collect_trajectory(
            agent, env, args.max_steps
        )
        
        # Update agent
        policy_loss, value_loss = agent.update(
            states, actions, rewards, next_states, terminated
        )
        
        # Track metrics
        episode_reward = rewards.sum().item()
        episode_rewards.append(episode_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
        # Periodic evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_metrics = evaluate_agent(agent, env)
            avg_reward = np.mean(episode_rewards[-args.eval_freq:])
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{args.num_episodes}")
            print(f"{'='*60}")
            print(f"Training:")
            print(f"  Avg Reward (last {args.eval_freq}): {avg_reward:.2f}")
            print(f"  Avg Delay (last {args.eval_freq}): {-avg_reward:.2f}")
            print(f"  Policy Loss: {policy_loss:.4f}")
            print(f"  Value Loss: {value_loss:.4f}")
            print(f"\nEvaluation:")
            print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Mean Delay: {-eval_metrics['mean_reward']:.2f}")
            print(f"  Mean Episode Length: {eval_metrics['mean_length']:.1f}")
            print(f"  Mean Completion Rate: {eval_metrics['mean_completion_rate']:.2%}")
            print(f"{'='*60}\n")
            
        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f'agent_episode_{episode + 1}.pt'
            )
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    # Save final model
    final_path = os.path.join(args.save_dir, 'agent_final.pt')
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_metrics = evaluate_agent(agent, env, num_eval_episodes=50)
    print(f"\nFinal Evaluation Results:")
    print(f"  Mean Reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"  Mean Delay: {-final_metrics['mean_reward']:.2f}")
    print(f"  Mean Episode Length: {final_metrics['mean_length']:.1f}")
    print(f"  Mean Completion Rate: {final_metrics['mean_completion_rate']:.2%}")
    
    return agent, episode_rewards, policy_losses, value_losses


if __name__ == '__main__':
    args = parse_args()
    train(args)
