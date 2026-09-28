"""Experience replay buffer (Session 2, slide 11; TD3 Task 1.3).

Same role as the deque in TD3: store transitions (s, a, r, s', done) and sample
uniform random mini-batches. Implemented as a preallocated numpy ring buffer:
- writing into the arrays copies the observation, so the env's in-place
  rewriting of its observation rows cannot corrupt stored transitions;
- sampling does not copy the whole buffer every step (TD3's list(deque));
- it has its own seeded random generator, so the env's randomness (drawn from
  the global np.random) is not shifted by the agent's sampling.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_shape, seed=0):
        self.capacity = capacity
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0     # next slot to write (oldest transition once full)
        self.size = 0    # number of stored transitions
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done):
        """Store one transition; overwrite the oldest one when full (like deque(maxlen))."""
        self.states[self.pos] = state            # array assignment = copy
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Uniform mini-batch without replacement (like random.sample in TD3)."""
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

    def __len__(self):
        return self.size
