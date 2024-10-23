import random
import matplotlib.pyplot as plt

# --- Binary Bandit ---
class BinaryBandit:
    def __init__(self):
        self.arms = 2  # Number of arms is fixed to 2 for binary bandit
        self.p = [random.random() for _ in range(self.arms)]  # Random success probabilities

    def get_actions(self):
        return list(range(self.arms))  # Actions: 0 or 1

    def get_reward(self, action):
        return 1 if random.random() < self.p[action] else 0  # 1 for success, 0 for failure

# --- Epsilon-Greedy Agent for Binary Bandit ---
def epsilon_greedy_agent(bandit, epsilon, steps):
    num_actions = bandit.arms
    Q_values = [0.0] * num_actions  # Initialize estimated rewards (Q-values)
    action_counts = [0] * num_actions  # Track how many times each action is selected
    rewards_history = []  # Store all rewards over steps
    avg_rewards_history = [0.0]  # Track the running average reward

    for t in range(1, steps + 1):
        # Epsilon-greedy decision
        if random.random() > epsilon:
            action = Q_values.index(max(Q_values))  # Exploit: choose best action
        else:
            action = random.choice(bandit.get_actions())  # Explore: choose a random action

        # Get the reward for the chosen action
        reward = bandit.get_reward(action)
        rewards_history.append(reward)
        action_counts[action] += 1

        # Update estimated Q-value incrementally
        Q_values[action] += (reward - Q_values[action]) / action_counts[action]

        # Update running average reward
        avg_rewards_history.append(avg_rewards_history[t - 1] + (reward - avg_rewards_history[t - 1]) / t)

    # Display action counts
    plt.bar(["Action 1", "Action 2"], action_counts)
    plt.title("Number of Times Each Action Taken")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.show()

    return Q_values, avg_rewards_history, rewards_history

# --- Simulation for Bandit A ---
random.seed(3)
bandit_A = BinaryBandit()
Q_values_A, avg_rewards_A, rewards_A = epsilon_greedy_agent(bandit_A, epsilon=0.1, steps=2000)

# Print results for Bandit A
print("****************** RESULTS FOR BANDIT A *************************")
print(f"Observed Average Reward for Action 1: {Q_values_A[0]}")
print(f"Observed Average Reward for Action 2: {Q_values_A[1]}")
print(f"Actual Reward Probability for Action 1: {bandit_A.p[0]}")
print(f"Actual Reward Probability for Action 2: {bandit_A.p[1]}")
print("***************************************************************")

# Plot the reward trends for Bandit A
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(avg_rewards_A)
ax1.set_title("Average Reward vs Iteration for Bandit A")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(rewards_A)
ax2.set_title("Reward per Iteration for Bandit A")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
plt.show()

# --- Simulation for Bandit B ---
random.seed(9)
bandit_B = BinaryBandit()
Q_values_B, avg_rewards_B, rewards_B = epsilon_greedy_agent(bandit_B, epsilon=0.1, steps=2000)

# Print results for Bandit B
print("****************** RESULTS FOR BANDIT B *************************")
print(f"Observed Average Reward for Action 1: {Q_values_B[0]}")
print(f"Observed Average Reward for Action 2: {Q_values_B[1]}")
print(f"Actual Reward Probability for Action 1: {bandit_B.p[0]}")
print(f"Actual Reward Probability for Action 2: {bandit_B.p[1]}")
print("***************************************************************")

# Plot the reward trends for Bandit B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(avg_rewards_B)
ax1.set_title("Average Reward vs Iteration for Bandit B")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(rewards_B)
ax2.set_title("Reward per Iteration for Bandit B")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
plt.show()
