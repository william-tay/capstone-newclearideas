import random
from collections import deque
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils

#env
class NetworkResilienceEnv:
    """
    Simple environment for DQN:
    - Start from base_graph.
    - Add a fixed number of new nodes.
    - For each new node, add a fixed number of edges to existing nodes.
    - Reward = change in resilience - cost.
    """

    def __init__(
        self,
        base_graph: nx.Graph,
        n_new_nodes=3,
        edges_per_new_node=2,
        n_fail_samples=5,
        lambda_cost=0.1,
        node_cost=1.0,
        edge_cost=0.2,
        n_max=50,
    ):
        self.base_graph = base_graph.copy()
        self.n_new_nodes = n_new_nodes
        self.edges_per_new_node = edges_per_new_node
        self.n_fail_samples = n_fail_samples
        self.lambda_cost = lambda_cost
        self.node_cost = node_cost
        self.edge_cost = edge_cost
        self.n_max = n_max

        assert self.base_graph.number_of_nodes() <= n_max

        self.num_actions = n_max

        self.obs_dim = self.n_max * self.n_max + self.n_max

        # Internal state
        self.G = None
        self.current_new_node_index = None
        self.edges_added_for_current_new = None
        self.current_new_node = None
        self.initial_resilience = None
        self.total_cost = None

    def _resilience(self, G: nx.Graph) -> float:
        """
        Simple resilience metric:
        - For each trial, remove ~10% of nodes at random.
        - Compute size of largest connected component afterward.
        - Return average over trials.
        """
        n = G.number_of_nodes()
        if n == 0:
            return 0.0

        failures_per_trial = max(1, n // 10)
        sizes = []
        nodes = list(G.nodes())
        for _ in range(self.n_fail_samples):
            failed = np.random.choice(nodes, size=failures_per_trial, replace=False)
            H = G.copy()
            H.remove_nodes_from(failed)
            if len(H) == 0:
                sizes.append(0)
            else:
                cc_sizes = [len(c) for c in nx.connected_components(H)]
                sizes.append(max(cc_sizes))
        return float(np.mean(sizes))

    def _get_obs(self):
        """
        Build padded adjacency matrix + degree vector.
        Nodes are remapped to [0, n_current-1].
        """
        G = self.G
        mapping = {node: i for i, node in enumerate(G.nodes())}

        A = np.zeros((self.n_max, self.n_max), dtype=np.float32)
        for u, v in G.edges():
            i, j = mapping[u], mapping[v]
            A[i, j] = 1.0
            A[j, i] = 1.0

        deg = np.zeros(self.n_max, dtype=np.float32)
        for node, d in G.degree():
            deg[mapping[node]] = d

        obs = np.concatenate([A.flatten(), deg], axis=0)
        return obs

    def reset(self):
        self.G = self.base_graph.copy()
        self.current_new_node_index = 0
        self.edges_added_for_current_new = 0
        self.total_cost = 0.0

        # Baseline resilience on graph
        base_resilience = self._resilience(self.G)

        # Add first new node
        self.current_new_node = max(self.G.nodes(), default=-1) + 1
        self.G.add_node(self.current_new_node)
        self.total_cost += self.node_cost

        # New baseline after node added
        self.initial_resilience = self._resilience(self.G)

        obs = self._get_obs()
        info = {"resilience": self.initial_resilience, "base_resilience": base_resilience}
        return obs.astype(np.float32), info

    def step(self, action: int):
        """
        Take an action:
        - action is an integer in [0, num_actions-1]
        :return: next_obs, reward, done, info
        """
        existing_nodes = list(self.G.nodes())
        n_current = len(existing_nodes)

        # Invalid action: picked index >= current number of nodes
        if action >= n_current:
            # Penalize and do nothing
            reward = -1.0
            obs = self._get_obs()
            info = {
                "resilience": self.initial_resilience,
                "total_cost": self.total_cost,
                "invalid_action": True,
            }
            done = False
            return obs.astype(np.float32), reward, done, info

        target_node = existing_nodes[action]

        # Add edge from current_new_node to target_node (if not already present)
        if not self.G.has_edge(self.current_new_node, target_node):
            self.G.add_edge(self.current_new_node, target_node)
            self.total_cost += self.edge_cost
            self.edges_added_for_current_new += 1

        # Compute resilience and shaped reward
        new_resilience = self._resilience(self.G)
        delta_R = new_resilience - self.initial_resilience
        step_cost = self.edge_cost
        reward = delta_R - self.lambda_cost * step_cost

        # Update baseline resilience for next step
        self.initial_resilience = new_resilience

        done = False

        # If we've added enough edges to this new node:
        if self.edges_added_for_current_new >= self.edges_per_new_node:
            self.current_new_node_index += 1
            self.edges_added_for_current_new = 0

            if self.current_new_node_index >= self.n_new_nodes:
                # No more new nodes -> episode ends
                done = True
            else:
                # Start next new node
                self.current_new_node = max(self.G.nodes()) + 1
                self.G.add_node(self.current_new_node)
                self.total_cost += self.node_cost
                # Reset baseline resilience after adding node (no edges yet)
                self.initial_resilience = self._resilience(self.G)

        obs = self._get_obs()
        info = {"resilience": new_resilience, "total_cost": self.total_cost}
        return obs.astype(np.float32), reward, done, info

#dqn
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

#training
def train_dqn(
    env: NetworkResilienceEnv,
    num_episodes=500,
    replay_capacity=10000,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_episodes=300,
    target_update_every=50,
    max_steps_per_episode=100,
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    state_dim = env.obs_dim
    action_dim = env.num_actions

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)

    epsilon = initial_epsilon

    # Linear epsilon decay
    def get_epsilon(episode):
        if episode >= epsilon_decay_episodes:
            return final_epsilon
        frac = episode / float(epsilon_decay_episodes)
        return initial_epsilon + frac * (final_epsilon - initial_epsilon)

    all_episode_rewards = []
    all_episode_losses = []


    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_losses = []

        for t in range(max_steps_per_episode):
            # Epsilon-greedy action selection
            epsilon = get_epsilon(episode)
            if random.random() < epsilon:
                action = random.randrange(env.num_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(s)
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Update DQN
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_dqn_loss(policy_net, target_net, batch, gamma, device)

                optimizer.zero_grad()
                loss.backward()
                nn_utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                # track loss for this episode
                episode_losses.append(loss.item())

            if done:
                break
        # After the episode finishes
        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        all_episode_rewards.append(episode_reward)
        all_episode_losses.append(mean_loss)

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Loss: {mean_loss:.4f} | "
            f"Epsilon: {epsilon:.3f}"
        )

        # Target network update
        if (episode + 1) % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
    mean_r, std_r = evaluate_policy(env, trained_policy, n_episodes=30, device=device)
    print("Final evaluation:", mean_r, std_r)

    return policy_net, target_net, all_episode_rewards, all_episode_losses


loss_fn = nn.MSELoss()

def compute_dqn_loss(policy_net, target_net, batch, gamma, device):
    (
        states_b,
        actions_b,
        rewards_b,
        next_states_b,
        dones_b,
    ) = batch

    # Convert to tensors
    states_b = torch.tensor(states_b, dtype=torch.float32, device=device)
    actions_b = torch.tensor(actions_b, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_b = torch.tensor(rewards_b, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_b = torch.tensor(next_states_b, dtype=torch.float32, device=device)
    dones_b = torch.tensor(dones_b, dtype=torch.float32, device=device).unsqueeze(1)

    # Current Q(s,a) from policy net
    q_values = policy_net(states_b).gather(1, actions_b)

    # Target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states_b).max(dim=1, keepdim=True)[0]
        target_q_values = rewards_b + gamma * (1.0 - dones_b) * next_q_values

    loss_fn = nn.SmoothL1Loss()

    # inside update step:
    loss = loss_fn(q_values, target_q_values)

    return loss


import numpy as np
import torch


def evaluate_policy(env, policy_net, n_episodes=20, device="cpu", verbose=True):
    """
    Runs the policy greedily for n_episodes and measures performance.
    :returns: mean_reward, std_reward
    """
    policy_net.eval()  # inference mode

    total_rewards = []

    for ep in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Convert state to tensor
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Greedy action
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            # Step the env
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

        if verbose:
            print(f"Eval Episode {ep + 1}/{n_episodes} | Reward: {episode_reward:.3f}")

    mean_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))

    if verbose:
        print(f"\n=== Evaluation Summary ===")
        print(f"Mean Reward: {mean_reward:.3f}")
        print(f"Std Reward:  {std_reward:.3f}\n")

    policy_net.train()  # back to training mode
    return mean_reward, std_reward

if __name__ == "__main__":
    import networkx as nx

    # 1) Load your graph from an edge list file
    raw_G = nx.read_edgelist(
        "../tech-as-topology/tech-as-topology-visual.edges",
        nodetype=int  # ensure nodes are integers
    )

    # (Optional but safe) Relabel nodes to a compact 0..N-1 range
    base_G = nx.convert_node_labels_to_integers(raw_G, label_attribute="old_id")
    print("Loaded graph with", base_G.number_of_nodes(), "nodes and", base_G.number_of_edges(), "edges")

    # 2) Set n_max >= base number of nodes + how many new nodes you want
    n_max = base_G.number_of_nodes() + 20  # allow 20 new nodes, adjust as you like

    env = NetworkResilienceEnv(
        base_graph=base_G,
        n_new_nodes=3,          # how many new nodes agent can add
        edges_per_new_node=2,   # edges per new node
        n_fail_samples=5,
        lambda_cost=0.1,
        node_cost=1.0,
        edge_cost=0.2,
        n_max=n_max,
    )

    trained_policy, trained_target, rewards_history, loss_history = train_dqn(
        env,
        num_episodes=300,
        replay_capacity=5000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_episodes=200,
        target_update_every=25,
        max_steps_per_episode=100,
    )

