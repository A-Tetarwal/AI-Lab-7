import random
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms):
        self.arms = arms  # Number of arms
        self.expRewards = [7.0] * arms  # Initialize expected rewards

    def get_actions(self):
        return list(range(self.arms))  # Actions: 0 to arms-1

    def get_reward(self, action):
        # Incrementing the rewards at each step
        for i in range(self.arms):
            self.expRewards[i] += random.gauss(0, 0.01)  # Random walk for each arm
        return self.expRewards[action]  # Return the reward of the selected action

# --- Epsilon-Greedy Agent ---
def epsilon_greedy_agent(bandit, epsilon, steps, alpha):
    num_actions = bandit.arms
    Q_values = [0.0] * num_actions  # Initialize estimated rewards (Q-values)
    action_counts = [0] * num_actions  # Track the number of times each action is chosen
    rewards_history = []  # Store all rewards over steps
    avg_rewards_history = [0.0]  # Store the running average of rewards

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

        # Update estimated Q-value using incremental formula
        Q_values[action] += alpha * (reward - Q_values[action])

        # Update running average reward
        avg_rewards_history.append(avg_rewards_history[t - 1] + (reward - avg_rewards_history[t - 1]) / t)

    # Plot the action counts
    plt.bar(range(1, num_actions + 1), action_counts)
    plt.title("Number of Times Each Action Taken")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.show()

    return Q_values, avg_rewards_history, rewards_history

# --- Simulation for Bandit A ---
random.seed(3)
bandit_A = Bandit(arms=10)  # Initialize bandit with 10 arms
Q_values_A, avg_rewards_A, rewards_A = epsilon_greedy_agent(bandit_A, epsilon=0.4, steps=10000, alpha=0.01)

# Print results for Bandit A
print("****************** RESULTS FOR BANDIT A *************************")
for i in range(1, bandit_A.arms + 1):
    print(f"Observed Average Reward for Action {i}: {Q_values_A[i-1]}")
    print(f"Actual Reward for Action {i}: {bandit_A.expRewards[i-1]}")
    print("----------------------------------------------------------------------------------")

# Plot reward trends for Bandit A
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(avg_rewards_A)
ax1.set_title("Average Rewards vs Iteration for Bandit A")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(rewards_A)
ax2.set_title("Reward per Iteration for Bandit A")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
fig.suptitle("Modified Epsilon Greedy Policy")
plt.show()
