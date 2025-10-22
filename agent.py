

import numpy as np
import random 

class QLearningAgent:

    def __init__(self, state_space, action_space_size, alpha=1.0, gamma=0.9, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.state_space = state_space
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self._initialize_q_table()

        print(f"Q-Learning Agent initialized with {len(self.state_space)} states and {self.action_space_size} actions.")

    def _initialize_q_table(self):

        for state in self.state_space:
            for action in range(self.action_space_size):
                self.q_table[(state, action)] = 0.0

    def choose_action(self, state):

        if state not in self.state_space:
            temp_state_space = list(self.state_space)
            temp_state_space.append(state)
            self.state_space = tuple(temp_state_space)
            for action in range(self.action_space_size):
                if (state, action) not in self.q_table:
                    self.q_table[(state, action)] = 0.0

        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)
        else:
            q_values_for_state = [self.q_table.get((state, action), 0.0) for action in range(self.action_space_size)]
            
            max_q = np.max(q_values_for_state)
            best_actions = [i for i, q in enumerate(q_values_for_state) if q == max_q]
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):

        current_q_value = self.q_table.get((state, action), 0.0)
        
        max_q_next_state = 0.0
        if next_state is not None:
            if next_state not in self.state_space:
                temp_state_space = list(self.state_space)
                temp_state_space.append(next_state)
                self.state_space = tuple(temp_state_space)
                for a_prime in range(self.action_space_size):
                    if (next_state, a_prime) not in self.q_table:
                        self.q_table[(next_state, a_prime)] = 0.0

            q_values_for_next_state = [self.q_table.get((next_state, a_prime), 0.0) for a_prime in range(self.action_space_size)]
            max_q_next_state = np.max(q_values_for_next_state)
        
        td_target = reward + self.gamma * max_q_next_state
        td_error = td_target - current_q_value
        
        self.q_table[(state, action)] = current_q_value + self.alpha * td_error

    def decay_epsilon(self):

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)