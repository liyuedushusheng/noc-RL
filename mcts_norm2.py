#递归迭代
import numpy as np

from env2 import NOC_env
from BookSim_wrapper import BookSimLatencySimulator
import math
from copy import deepcopy
from tqdm import trange
from pathlib import Path
import concurrent.futures

max_value = -np.inf
save_dir = Path("./result")
save_dir.mkdir(parents=True, exist_ok=True)

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, is_root=False):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        if is_root:
            self.first_position_list = self.game.get_first_step()
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        
        self.children = []
        
        self.expandable_moves = game.get_valid_moves(state)
        
        self.visit_count = 0
        self.max_value = -float('inf')
        self.max_value_norm = 0
        self.value_sum = 0
        self.u_value_sum = 0
        
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            #print(f"child pos: {child.action_taken}, UCB: {ucb:.4f}")
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        print(f"select node: {best_child.action_taken}")
                
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.max_value_norm
        ucb = q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
        ucb_q = q_value
        ucb_explo = self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
        print(f"child pos: {child.action_taken}, UCB: {ucb:.4f}, Q: {ucb_q:.4f}, Explore: {ucb_explo:.4f}")
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self,all_node):
        if hasattr(self, "first_position_list"):
            print(self.first_position_list)
            action = self.first_position_list[len(self.children)]
            self.expandable_moves[action] = 0
        else:
            action = np.random.choice(np.where(self.expandable_moves == 1)[0])
            self.expandable_moves[action] = 0
        
        child_state = deepcopy(self.state)
        child_state = self.game.get_next_state(child_state, action)

        #print(child_state)
        
        child = Node(self.game, self.args, child_state, self, action)


        self.children.append(child)
        all_node.append(child)
        return child
    
    def simulate(self, search, id):
        value, is_terminal = self.game.get_value_and_terminated(self.state, id)
        global max_value

        if is_terminal:
            if value >= max_value:
                max_value = value
                np.save(save_dir / f"max_value_{max_value}_after.npy", np.array(self.state[1]))
                return value, self.state, f"Search {search}: value = {max_value}\n"
            return value, self.state, None
        
        rollout_state = deepcopy(self.state)
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, id)
            if is_terminal:
                if value >= max_value:
                    max_value = value
                    np.save(save_dir / f"max_value_{max_value}.npy", np.array(rollout_state[1]))
                    return value, np.array(rollout_state[1]), f"Search {search}: value = {max_value}\n"
                return value, np.array(rollout_state[1]), None
            
    def backpropagate(self, value):
        self.value_sum += reward_to_norm(value)
        self.u_value_sum += value
        self.visit_count += 1
        self.max_value = max(self.max_value, value)

        if self.parent is not None:
            self.parent.backpropagate(value)

global_max = -float('inf')
global_min = float('inf')

def reward_to_norm(reward):
    global global_max, global_min
    if global_max == -float('inf'):
        global_max = reward
    if global_min == float('inf'):
        global_min = reward

    if reward > global_max:
        global_max = reward
    if reward < global_min:
        global_min = reward

    if global_max == global_min:
        return 0.5
    return (reward - global_min) / (global_max - global_min)

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, state,executor):
        all_node = []
        root = Node(self.game, self.args, state, is_root=True)
        logs=[]
        search = 0
        while search < self.args['num_searches']:
            node = root

            
            # while node.is_fully_expanded():
            #     node = node.select()
            max_depth = self.args['depth_limit'] + search // self.args['leaf_searchs']

            while len(node.children) == self.args['leaf_limit'] and node.depth < max_depth:
                node = node.select()

            #value, is_terminal = self.game.get_value_and_terminated(node.state)
            if not node.is_fully_expanded():
                print("is terminal")
            
            if not is_terminal and node.depth < max_depth:
                node = node.expand(all_node)
            
            print(f"[Search {search}] State: {node.state}")

            value, res = node.simulate(search)
            with open(save_dir / "search_log.txt", "a") as f:
                f.write(f"Search {search}: value = {value}\n")
                #np.save(save_dir / f"res_{value}_in_sim{search}.npy", res)
                
            node.backpropagate(value)
            for node in all_node:
                node.max_value_norm = reward_to_norm(node.max_value)
            
            search += 1

            if search > 0 and search % self.args['leaf_searchs'] == 0:
                if not is_terminal:
                    best_child = max(root.children, key=lambda c: c.max_value_norm)
                    print(f"Switching root to best child: action={best_child.action_taken}, value_norm={best_child.max_value_norm:.4f}")
                    
                    best_child.parent = None
                    root = best_child
            futures = [executor.submit(node.simulate, search, id) for id in range(5)]
            for future in concurrent.futures.as_completed(futures):
                value, res, log = future.result()
                if log:
                    logs.append(log)
                node.backpropagate(value)

        with open(save_dir / "search_log.txt", "a") as f:
            for log in logs:
                f.write(log)

        action_probs = np.zeros(self.game.action_size)
        values = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
            values[child.action_taken] = child.value_sum / child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs, values

simulator_id = 1
simulator = BookSimLatencySimulator(
    config_addr="./config_VGG.json",
    simulator_id=simulator_id,
    simulator_path=f"./ISAAC_NOPU_MESH_FINAL_{simulator_id}"
)
env = NOC_env(simulator)

args = {
    'C': math.sqrt(2),
    'num_searches': 5000,
    'leaf_searchs': 20,
    'leaf_limit': 5,
    'depth_limit':5
}

mcts = MCTS(env, args)

state = env.get_initial_state()

# position = [16, 25, 27]
# for i in range(len(position)):
#     x, y = env._convert_to_coordinates(position[i])
#     state[0][x, y] = i

# state = (state[0], position)
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:：
    while True:
        mcts_probs, values = mcts.search(state,executor)
        action = np.argmax(mcts_probs)
        print(f"state={state}, mcts_probs={mcts_probs}, values={values}, action={action}")    
        state = env.get_next_state(state, action)
    
        value, is_terminal = env.get_value_and_terminated(state)
    
        if is_terminal:
            print(value)
            break



# --------------------------------------

# state=(array([[ 8., 15., 16., 12.,  9.],
#        [ 3., 17., 10., 13.,  5.],
#        [18.,  0.,  1., 14., 19.],
#        [ 4.,  6., 20., 21., 23.],
#        [11.,  7.,  2., 22., -1.]]), [11, 12, 22, 5, 15, 9, 16, 21, 0, 4, 7, 20, 3, 8, 13, 1, 2, 6, 10, 14, 17, 18, 23, 19]), mcts_probs=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  1.], values=[     0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0.      0.      0.      0.
#       0.      0.      0.      0.      0.      0. -24872.], action=24
# -24872