"""DQN agent with target network and experience replay (Session 2; TD3 Part 1).

Ported from TD3's DQNAgent. Changes vs TD3 are marked "Change vs TD3":
1. Replay buffer: numpy ring buffer (rl_trading/replay.py) instead of a deque.
2. Own seeded random generators for exploration and replay sampling.
3. choose_action_eval(): greedy action without exploration, as required by
   the course's evaluate_agent.
4. The target computation is a separate method, so Double DQN can override
   only that part (TD3 Part 3 overrides the whole train_step).
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_trading.networks import DQNetwork
from rl_trading.replay import ReplayBuffer


class DQNAgent:
    def __init__(self, state_size, n_actions, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.99,
                 epsilon_min=0.01, batch_size=64, target_update_freq=250, replay_buffer_size=50000,
                 hidden_size=128, seed=0):
        """DQN agent. Network weights are initialised from torch's global seed
        (call torch.manual_seed before creating the agent)."""

        # DQN hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Environment parameters
        self.state_size = state_size
        self.n_actions = n_actions

        # Everything needed to rebuild the agent (saved with the weights)
        self.config = dict(state_size=state_size, n_actions=n_actions, lr=lr, gamma=gamma,
                           epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                           batch_size=batch_size, target_update_freq=target_update_freq,
                           replay_buffer_size=replay_buffer_size, hidden_size=hidden_size, seed=seed)

        # Device configuration (as in TD3: MPS disabled, small MLPs are faster on CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Q-network and target Q-network (independent copy, same initial weights)
        self.q_network = DQNetwork(state_size, n_actions, hidden_size=hidden_size).to(self.device)
        self.target_q_network = deepcopy(self.q_network).to(self.device)
        self.target_q_network.eval()

        # Step counter for target network updates
        self.step_count = 0

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # Change vs TD3 (2): separate seeded generators for exploration and replay
        # sampling, independent of the global np.random that the env uses.
        explore_seed, replay_seed = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(explore_seed)

        # Change vs TD3 (1): numpy ring buffer, copies observations on insert
        self.replay_buffer = ReplayBuffer(replay_buffer_size, (state_size,), seed=replay_seed)

    def choose_action(self, state):
        """Epsilon-greedy action selection (training)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))  # Random action
        return self.choose_action_eval(state)

    def choose_action_eval(self, state):
        """Greedy action, no exploration (Change vs TD3 (3): used by evaluate_agent)."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer (done = terminated, not truncated)."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def compute_targets(self, reward_tensor, next_state_tensor, done_tensor):
        """DQN target y = r + (1 - done) * γ * max_a' Q_θ⁻(s', a') (Change vs TD3 (4): own method)."""
        next_q_values = self.target_q_network(next_state_tensor)   # (batch_size, n_actions)
        max_next_q = next_q_values.max(dim=1).values                # (batch_size,)
        return reward_tensor + (1.0 - done_tensor) * self.gamma * max_next_q

    def train_step(self):
        """One DQN update from the replay buffer using the target network."""

        # If not enough experiences, return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        state_tensor = torch.as_tensor(states, device=self.device)            # (batch_size, state_size)
        next_state_tensor = torch.as_tensor(next_states, device=self.device)  # (batch_size, state_size)
        action_tensor = torch.as_tensor(actions, device=self.device)          # (batch_size,)
        reward_tensor = torch.as_tensor(rewards, device=self.device)          # (batch_size,)
        done_tensor = torch.as_tensor(dones, device=self.device)              # (batch_size,)

        # Get current Q-value for the action taken
        current_q = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1))  # (batch_size, 1)
        current_q = current_q.squeeze(-1)                                               # (batch_size,)

        # Calculate target Q-value (no gradient through the target)
        with torch.no_grad():
            target_q = self.compute_targets(reward_tensor, next_state_tensor, done_tensor)

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment step counter and update target network if needed
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath):
        """Save the network, optimizer, exploration rate and config."""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
        }, filepath)

    def load_model(self, filepath):
        """Load the network, optimizer and exploration rate (target network synced to the loaded weights)."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class DoubleDQNAgent(DQNAgent):
    """Double DQN (Session 2, slides 14 to 16; TD3 Part 3).

    Only the target changes: the online network selects the next action, the
    target network evaluates it, which reduces the overestimation bias of max:
        y = r + (1 - done) * γ * Q_θ⁻(s', argmax_a' Q_θ(s', a'))
    """

    def compute_targets(self, reward_tensor, next_state_tensor, done_tensor):
        next_actions = torch.argmax(self.q_network(next_state_tensor), dim=1, keepdim=True)    # online selects
        next_q = self.target_q_network(next_state_tensor).gather(1, next_actions).squeeze(-1)  # target evaluates
        return reward_tensor + (1.0 - done_tensor) * self.gamma * next_q


def load_agent(checkpoint_path):
    """Rebuild an agent from a saved checkpoint (config + weights), greedy (epsilon 0)."""
    checkpoint = torch.load(checkpoint_path)
    agent = DQNAgent(**dict(checkpoint['config'], replay_buffer_size=1))  # no buffer needed for evaluation
    agent.load_model(checkpoint_path)
    agent.epsilon = 0.0
    return agent


class EnsembleAgent:
    """Greedy agent that averages the Q-values of several trained networks.

    Each network memorises different noise; averaging keeps what they share.
    """

    def __init__(self, agents):
        self.agents = agents
        self.device = agents[0].device
        self.q_network = self  # so diagnostics can call agent.q_network(states)

    def __call__(self, state_tensor):
        with torch.no_grad():
            return torch.stack([a.q_network(state_tensor) for a in self.agents]).mean(dim=0)

    def choose_action_eval(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.argmax(self(state_tensor)).item()


def save_ensemble(checkpoint_paths, path, metadata):
    """Store several trained networks (weights + config, no optimizer state) in one file."""
    members = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path)
        members.append({'config': checkpoint['config'], 'model_state_dict': checkpoint['model_state_dict']})
    torch.save({'members': members, 'metadata': metadata}, path)


def load_ensemble(path):
    """Load a file written by save_ensemble; returns (EnsembleAgent, metadata)."""
    checkpoint = torch.load(path)
    agents = []
    for member in checkpoint['members']:
        agent = DQNAgent(**dict(member['config'], replay_buffer_size=1))
        agent.q_network.load_state_dict(member['model_state_dict'])
        agent.q_network.eval()
        agent.epsilon = 0.0
        agents.append(agent)
    return EnsembleAgent(agents), checkpoint['metadata']
