import gymnasium as gym
import random

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

random.seed(0)

print("## Frozen Lake ##")

action2string = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

def play_episode(env, policy=None):
    state = env.reset(seed=0)[0]
    done = False
    total_reward = 0
    states = [state]
    actions = []
    while not done:
        if policy is None:
            action = random.randint(0, 3)
        else:
            action = policy[state]

        actions.append(action)
        state, reward, done, _, _ = env.step(action)
        states.append(state)
        total_reward += reward

    return states, actions, total_reward


###### Task 1 ######
print(f"\n ### TASK 1 ### ")

count = 0
while True:
    s, a, r = play_episode(env)
    count += 1
    if r > 0:
        break

print(f"Converged after {count} episodes.")
print(f"Random policy took {len(s)} steps.")
print(f"Visited states \t {s}")
print(f"Took actions \t {a}")



policy = {}
for i, v in enumerate(s[:-1]):
    policy[v] = a[i]
print("Improving policy to:", policy)

s, a, r = play_episode(env, policy)
if r > 0:
    print(f"Success: New policy took {len(s)} steps.")
else:
    print("New policy failed!")




###### Task 2 ######
print(f"\n ### TASK 2 ### ")

env_8x8 = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")

count = 0
while True:
    s, a, r = play_episode(env_8x8)
    count += 1
    if r > 0:
        break

print(f"Converged after {count} episodes.")
print(f"Took {len(s)} steps.")

policy_8x8 = {}
for i, v in enumerate(s[:-1]):
    policy_8x8[v] = a[i]
print("Improving policy to:", policy_8x8)

s, a, r = play_episode(env_8x8, policy_8x8)
if r > 0:
    print(f"Success: New policy took {len(s)} steps.")
else:
    print("New policy failed!")


###### Task 3 ######
print(f"\n ### TASK 3 ### ")
env_slippery = gym.make("FrozenLake-v1")
try:
    s, a, r = play_episode(env_slippery, policy)
    if r > 0:
        print(f"Success: New policy took {len(s)} steps.")
    else:
        print("New policy failed!")

except KeyError as e:
    print("Failure! Policy is not defined for state:", e)

# The problem is that the policy from task 1 did not see all states,
# which means that the policy is not defined for all states.
# Since the environment is now slippery we can reach such a state and
# the algorithm crashes. 
# We can fix this by changing line 23 to:
#    if policy is None or state not in policy:
# In this case the policy falls back to random if state is not set
