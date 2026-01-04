"""
Operator Agent with Actor-Critic RL

This module implements the traffic operator agent that learns to provide
regional traffic information to optimize system-wide efficiency.

Based on the Actor-Critic architecture with N-step returns.
"""

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from typing import Tuple, List


class Policy(nn.Module):
    """Actor network that outputs action logits."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(OrderedDict([
            ('policy_layer_1', nn.Linear(state_dim, hidden_dim)),
            ('policy_activation_1', nn.ReLU()),
            ('policy_layer_2', nn.Linear(hidden_dim, hidden_dim)),
            ('policy_activation_2', nn.ReLU()),
            ('policy_layer_3', nn.Linear(hidden_dim, action_dim)),
        ]))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class Value(nn.Module):
    """Critic network that estimates state value V(s)."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(OrderedDict([
            ('value_layer_1', nn.Linear(state_dim, hidden_dim)),
            ('value_activation_1', nn.ReLU()),
            ('value_layer_2', nn.Linear(hidden_dim, hidden_dim)),
            ('value_activation_2', nn.ReLU()),
            ('value_layer_3', nn.Linear(hidden_dim, 1)),
        ]))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class OperatorAgent(nn.Module):
    """
    Actor-Critic agent for the traffic operator.
    
    The operator observes the full traffic state and selects regional
    information signals to optimize system-wide traffic efficiency.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 3e-4,
        max_episode_steps: int = 100,
        n_step: int = 10,
        hidden_dim: int = 256,
    ):
        """
        Initialize the operator agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions (regional signals)
            gamma: Discount factor
            lr: Learning rate
            max_episode_steps: Maximum steps per episode
            n_step: N for N-step returns
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.max_episode_steps = max_episode_steps
        self.n_step = n_step
        
        # Initialize networks
        self.policy = Policy(state_dim, action_dim, hidden_dim)
        self.value = Value(state_dim, hidden_dim)
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Precompute cumulative discount factors
        self.cum_gamma = torch.tensor([gamma ** i for i in range(max_episode_steps + 1)], dtype=torch.float32)
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False, valid_actions: List[int] = None) -> int:
        """
        Select action given state.
        
        Args:
            state: Current state tensor
            deterministic: If True, select argmax action; otherwise sample
            valid_actions: List of valid action indices (if None, all actions are valid)
            
        Returns:
            Selected action (integer)
        """
        with torch.no_grad():
            state = state.detach()
            logits = self.policy(state)
            action_probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Mask invalid actions if valid_actions is provided
            if valid_actions is not None:
                mask = torch.zeros_like(action_probs)
                mask[valid_actions] = 1.0
                action_probs = action_probs * mask
                # Renormalize
                if action_probs.sum() > 0:
                    action_probs = action_probs / action_probs.sum()
                else:
                    # If all masked out, default to uniform over valid actions
                    action_probs = mask / mask.sum()
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                action = torch.multinomial(action_probs, num_samples=1).item()
                
        return action
        
    def get_action_and_value(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value estimate.
        
        Used during training to collect trajectory data.
        
        Args:
            state: Current state tensor
            
        Returns:
            action: Selected action
            log_prob: Log probability of the action
            value: Value estimate V(s)
        """
        logits = self.policy(state)
        action_probs = torch.nn.functional.softmax(logits, dim=-1)
        action = torch.multinomial(action_probs, num_samples=1).item()
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_prob = log_probs[action]
        
        value = self.value(state)
        
        return action, log_prob, value
        
    def n_step_returns(
        self,
        n: int,
        rewards: torch.Tensor,
        next_state_values: torch.Tensor,
        terminated: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate N-step returns for each timestep.
        
        n_step_return_t = r_t + r_{t+1}*γ + ... + r_{t+n-1}*γ^{n-1} + V(s_{t+n})*γ^n
        
        Args:
            n: Number of steps for returns
            rewards: Rewards at each timestep (N, T, 1)
            next_state_values: Value estimates for next states (N, T, 1)
            terminated: Terminal state flags (N, T, 1)
            
        Returns:
            N-step returns (N, T, 1)
        """
        _, T, _ = rewards.shape
        n = min(n, T)
        
        # Apply discount factors to rewards
        gammas = self.cum_gamma[:T].unsqueeze(0).unsqueeze(-1)
        discounted_rewards = gammas * rewards
        
        # Pad rewards: 1 zero on left, n-1 zeros on right
        padded_rewards = torch.nn.functional.pad(
            discounted_rewards, (0, 0, 1, n - 1, 0, 0), value=0
        )
        
        # Calculate cumulative sum and extract n-step reward sums
        cumsum = torch.cumsum(padded_rewards, dim=1)
        reward_terms = cumsum[:, n:] - cumsum[:, :-n]
        reward_terms = reward_terms / gammas  # Undo initial discounting
        
        # Get values at timestep t+n (zero if terminated)
        zeroed_next_state_values = next_state_values * (~terminated)
        last_val = zeroed_next_state_values[:, -1:, :]
        padded_vals = torch.cat(
            [zeroed_next_state_values, last_val.repeat(1, n - 1, 1)], dim=1
        )
        v_t_plus_n = padded_vals[:, n-1:, :]
        
        # Discount values
        discounted_values = v_t_plus_n * self.cum_gamma[n].item()
        
        # Combine reward terms and discounted values
        n_step_returns = reward_terms + discounted_values
        return n_step_returns
        
    def value_loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value network (critic) loss.
        
        Args:
            states: States at timesteps t, t+1, ...
            rewards: Rewards at timesteps t, t+1, ...
            next_states: States at timesteps t+1, t+2, ...
            terminated: Terminal state flags
            
        Returns:
            MSE loss between value predictions and n-step returns
        """
        with torch.no_grad():
            next_state_values = self.value(next_states)
            n_step_returns = self.n_step_returns(
                self.n_step, rewards, next_state_values, terminated
            )
            
        predictions = self.value(states)
        loss = torch.nn.functional.mse_loss(predictions, n_step_returns)
        return loss
        
    def update_value(self):
        """Update value network parameters."""
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()
        
    def policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute policy network (actor) loss.
        
        Uses advantage = n_step_return - V(s) for policy gradient.
        
        Args:
            states: States at timesteps t, t+1, ...
            actions: Actions taken at each timestep
            rewards: Rewards at timesteps t, t+1, ...
            next_states: States at timesteps t+1, t+2, ...
            terminated: Terminal state flags
            
        Returns:
            Policy gradient loss
        """
        # Compute advantages
        with torch.no_grad():
            next_state_values = self.value(next_states)
            n_step_returns = self.n_step_returns(
                self.n_step, rewards, next_state_values, terminated
            )
            
        value_estimate = self.value(states)
        advantages = n_step_returns - value_estimate
        
        # Compute log probabilities of taken actions
        logits = self.policy(states)
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(dim=-1, index=actions)
        
        # Policy gradient loss
        loss = -(advantages.detach() * log_probs).mean()
        return loss
        
    def update_policy(self):
        """Update policy network parameters."""
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Update both policy and value networks.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            terminated: Batch of terminal flags
            
        Returns:
            policy_loss: Policy loss value
            value_loss: Value loss value
        """
        # Update value network
        v_loss = self.value_loss(states, rewards, next_states, terminated)
        v_loss.backward()
        self.update_value()
        
        # Update policy network
        p_loss = self.policy_loss(states, actions, rewards, next_states, terminated)
        p_loss.backward()
        self.update_policy()
        
        return p_loss.item(), v_loss.item()
        
    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


def collect_trajectory(
    agent: OperatorAgent,
    env,
    max_steps: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect a full trajectory from the environment.
    
    Args:
        agent: Operator agent
        env: Traffic environment
        max_steps: Maximum steps per episode
        
    Returns:
        states: States at each timestep (1, T, state_dim)
        actions: Actions at each timestep (1, T, 1)
        rewards: Rewards at each timestep (1, T, 1)
        next_states: Next states at each timestep (1, T, state_dim)
        terminated: Terminal flags (1, T, 1)
    """
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    states = []
    actions = []
    rewards = []
    next_states = []
    terminateds = []
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Get action from agent
        action = agent.get_action(state)
        
        # Step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # Store transition
        states.append(state)
        actions.append(torch.tensor(action, dtype=torch.long))
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        next_states.append(next_state)
        terminateds.append(torch.tensor(terminated, dtype=torch.bool))
        
        state = next_state
        done = terminated or truncated
        step += 1
        
    # Stack tensors and add batch dimension
    states = torch.stack(states).unsqueeze(0)
    actions = torch.stack(actions).unsqueeze(0).unsqueeze(-1)
    rewards = torch.stack(rewards).unsqueeze(0).unsqueeze(-1)
    next_states = torch.stack(next_states).unsqueeze(0)
    terminateds = torch.stack(terminateds).unsqueeze(0).unsqueeze(-1)
    
    return states, actions, rewards, next_states, terminateds
