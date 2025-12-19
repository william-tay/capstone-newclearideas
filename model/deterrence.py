import numpy as np
"""
It is the perceptions of the potential
aggressor that matter, not the actual prospects for victory or the
objectively measured consequences of an attack. Perceptions are the
dominant variable in deterrence success or failure

Deterrence succeeds, when it does, by creating a subjective perception in the minds of the leaders of the target state.
How can we model this perception in the other agent?

3 Keys to Deterrence:
- Level of Aggressor Motivation
- Clarity About the Object of Deterrence and Actions the Defender Will Take
- Aggressor Must Be Confident that Deterring State Has Capability and Will to Carry Out Threats

Matrix Approach?
[threaten, , ]
"""
class DeterrenceEnv:
    """
    Simple two-player zero-sum deterrence environment.

    State:
        - Represented by an integer tension level: 0 (low), 1 (medium), 2 (high).

    Actions (for both agent and opponent):
        - 0: Restrain   (de-escalatory move)
        - 1: Maintain   (status-quo)
        - 2: Escalate   (escalatory move)
    """

    def __init__(self, max_steps=50, seed=None):
        # Number of discrete tension levels (0, 1, 2)
        self.n_states = 3

        # Number of discrete actions for each player
        self.n_actions = 3          # agent actions
        self.n_opp_actions = 3      # opponent actions

        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Internal environment state
        self.tension = None
        self.steps = 0

    def reset(self):
        """
        Reset the environment to an initial state.
        Returns the initial state (tension level).
        """
        # Start at medium tension by default
        self.tension = 1
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """
        Return the current state representation.
        For now, this is just the tension level (0, 1, or 2).
        """
        return self.tension

    def step(self, agent_action, opp_action):
        """
        Take one step in the environment given the agent's action and the opponent's action.

        Args:
            agent_action (int): action index for the agent  (0, 1, 2)
            opp_action   (int): action index for the opponent (0, 1, 2)

        Returns:
            next_state (int): the next tension level (0, 1, 2)
            reward_agent (float): reward for the agent
            done (bool): whether the episode has terminated
            info (dict): auxiliary info (e.g., opponent reward)
        """
        self.steps += 1

        # --- Update tension based on joint action ---
        # If either side escalates, tension increases.
        if agent_action == 2 or opp_action == 2:
            self.tension = min(self.tension + 1, self.n_states - 1)
        # If both restrain, tension decreases.
        elif agent_action == 0 and opp_action == 0:
            self.tension = max(self.tension - 1, 0)
        # Else: tension stays the same.

        # --- Compute reward for the agent ---
        # You can tweak these numbers, but the structure encodes deterrence logic.
        if agent_action == 2 and opp_action == 2:
            # Mutual escalation – worst case for both
            reward_agent = -5.0

        elif agent_action == 2 and opp_action == 0:
            # Agent escalates while opponent restrains – short-term advantage
            reward_agent = 1.0

        elif agent_action == 0 and opp_action == 2:
            # Agent restrains while opponent escalates – deterrence failure
            reward_agent = -1.0

        elif agent_action == 0 and opp_action == 0:
            # Mutual restraint – stability is mildly good
            reward_agent = 0.1

        else:
            # All other combinations (e.g., maintain vs anything, or asymmetries that
            # don't clearly fit the above cases) – treat as roughly neutral.
            reward_agent = 0.0


        # Termination condition: either we hit max_steps or tension reaches max level
        done = (self.steps >= self.max_steps)

        next_state = self._get_state()

        return next_state, reward_agent, done



class MinimaxQAgent:
    def __init__(self, n_states, n_actions, n_opp_actions,
                 gamma=0.95, lr=0.1, eps=0.1):
        self.n_states = n_states
        """
        Number of possible environment states.
        This defines the size of the state space S.
        """
        self.n_actions = n_actions
        """
        Number of actions available to the agent (size of the action set A).
        """
        self.n_opp_actions = n_opp_actions
        """
        Number of actions available to the opponent (size of the opponent's action set O)
        """
        self.gamma = gamma
        """
        Discount factor γ.
        Values close to 0 make the agent short-sighted (prefers immediate rewards).
        Values close to 1 make the agent long-term oriented (future rewards matter more).
        """
        self.lr = lr
        """
        Learning rate η.
        Controls how much newly observed information updates existing Q-values.
        Values close to 1 make learning fast but potentially unstable.
        Values close to 0 make learning slow but more stable.
        """
        self.eps = eps
        """
        Exploration rate ε for ε-greedy policy.
        With probability ε the agent chooses a random action to encourage exploration.
        ε decays over time so the agent explores early but exploits learned policies later.
        """
        # Q[s, a, o], new array of given shape and type, filled with zeros.
        self.Q = np.zeros((n_states, n_actions, n_opp_actions))


    def _state_policy_minimax(self, s):
        """
        Compute pi*(s) = argmax_pi min_o sum_a pi(a) Q(s,a,o).
        This is a pure strategy minimax, not a mixed strategy with linear solver
        For more actions (or more reasonable results), use LP instead.
        """
        # naive approach: restrict to pure strategies for now
        # (this turns it into max_a min_o Q(s,a,o))
        q_sa = self.Q[s]  # shape (n_actions, n_opp_actions)
        worst_case_for_each_a = q_sa.min(axis=1)  # min over o
        best_action = np.argmax(worst_case_for_each_a)

        pi = np.zeros(self.n_actions)
        pi[best_action] = 1.0
        return pi

    def select_action(self, s):
        # epsilon-greedy over minimax policy
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        pi = self._state_policy_minimax(s)
        return np.random.choice(self.n_actions, p=pi)

    def update(self, s, a, o, r, s_next):
        # compute V(s_next) via minimax on next state
        pi_next = self._state_policy_minimax(s_next)
        # V(s') = max_pi min_o sum_a pi(a) Q(s',a,o)
        # with our pure-strategy simplification, this is:
        q_next = self.Q[s_next]  # (a, o)
        worst_case_for_each_a = q_next.min(axis=1)
        V_next = worst_case_for_each_a.max()

        td_target = r + self.gamma * V_next
        td_error = td_target - self.Q[s, a, o]
        self.Q[s, a, o] += self.lr * td_error


env = DeterrenceEnv(max_steps=100, seed=42)

agent = MinimaxQAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    n_opp_actions=env.n_opp_actions,
    gamma=0.95,
    lr=0.1,
    eps=0.1,
)

opponent = MinimaxQAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    n_opp_actions=env.n_opp_actions,
    gamma=0.95,
    lr=0.1,
    eps=0.1,
)

num_episodes = 1000

agent_action_freqs = []   # list of dicts
opp_action_freqs = []     # list of dicts

for episode in range(num_episodes):
    s = env.reset()
    done = False

    # Per-episode action counts
    episode_counts_agent = {0: 0, 1: 0, 2: 0}
    episode_counts_opp = {0: 0, 1: 0, 2: 0}
    steps = 0

    while not done:
        a = agent.select_action(s)
        o = opponent.select_action(s)

        episode_counts_agent[a] += 1
        episode_counts_opp[o] += 1
        steps += 1

        s_next, r, done = env.step(a, o)
        opp_r = -r

        agent.update(s, a, o, r, s_next)
        opponent.update(s, o, a, opp_r, s_next)

        s = s_next

        agent.eps = max(0.01, agent.eps * 0.9999)
        opponent.eps = max(0.01, opponent.eps * 0.9999)

    # Convert counts to frequencies for this episode
    agent_freq = {a: episode_counts_agent[a] / steps for a in episode_counts_agent}
    opp_freq   = {a: episode_counts_opp[a]   / steps for a in episode_counts_opp}

    agent_action_freqs.append(agent_freq)
    opp_action_freqs.append(opp_freq)

    if episode % 100 == 0:
        print(f"\nEpisode {episode}")
        print("Agent freqs:   ", agent_freq)
        print("Opponent freqs:", opp_freq)

# Average over the last 100 episodes
window = 100
start = num_episodes - window

avg_agent = {0: 0.0, 1: 0.0, 2: 0.0}
avg_opp   = {0: 0.0, 1: 0.0, 2: 0.0}

for ep in range(start, num_episodes):
    for a in avg_agent:
        avg_agent[a] += agent_action_freqs[ep][a] / window
        avg_opp[a]   += opp_action_freqs[ep][a]   / window

print("\nAverage frequencies over last 100 episodes:")
print("Agent:", avg_agent)
print("Opp:  ", avg_opp)




