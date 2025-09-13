import numpy as np
from env2 import NOC_env
from BookSim_wrapper import BookSimLatencySimulator
import math
from copy import deepcopy
from tqdm import trange
from pathlib import Path
import concurrent.futures
import time
max_value = -np.inf
save_dir = Path("./result")
save_dir.mkdir(parents=True, exist_ok=True)

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        
        self.visit_count = 0
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
            q_value = child.value_sum / child.visit_count
        ucb = q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
        ucb_q = q_value
        ucb_explo = self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
        print(f"child pos: {child.action_taken}, UCB: {ucb:.4f}, Q: {ucb_q:.4f}, Explore: {ucb_explo:.4f}")
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        child_state = deepcopy(self.state)
        child_state = self.game.get_next_state(child_state, action)

        print(child_state)
        
        child = Node(self.game, self.args, child_state, self, action)


        self.children.append(child)
        return child
    
    def simulate(self, search, parallel_nums=5):
        seeds = [int(time.time()*1000000) % 2**32 + i * 1000 + search for i in range(parallel_nums)]
        
        value, is_terminal = self.game.get_value_and_terminated(self.state,1) 
        
        global max_value

        if is_terminal:
            if value >= max_value:
                max_value = value
                np.save(save_dir / f"max_value_{max_value}_after.npy", np.array(self.state[1]))
                with open(save_dir / "max_search_log.txt", "a") as f:
                    f.write(f"Search {search}: value = {max_value}\n")
            return value, is_terminal
        #print(f"self.state:{self.state}")
        rollout_states = [deepcopy(self.state) for _ in range(parallel_nums)]
        while True:
            for i in range(parallel_nums):
                np.random.seed(seeds[i])
                valid_moves = self.game.get_valid_moves(rollout_states[i])
                action = np.random.choice(np.where(valid_moves == 1)[0])
                rollout_states[i] = self.game.get_next_state(rollout_states[i], action)
            
            #print(f"rollout_states_len:{len(rollout_states)}")
            #print(f"parallel_nums:{parallel_nums}")

            values, is_terminal = self.game.get_value_and_terminated(rollout_states,parallel_nums)
            #print(f"values:{values},type: {type(values)}")
            if is_terminal:
                rollout_state = None
                for i in range(parallel_nums):
                    #print(f"values[i]:{values[i]},type: {type(values[i])}")
                    if values[i] >= max_value:
                        max_value = values[i]
                        rollout_state=rollout_states[i]
                if rollout_state is not None:
                    np.save(save_dir / f"max_value_{max_value}.npy", np.array(rollout_state[1]))
                    with open(save_dir / "max_search_log.txt", "a") as f:
                        f.write(f"Search {search}: value = {max_value}\n")
                for i in range(parallel_nums):
                    print(f"values[{i}]: {values[i]}")
                    print(f"rollout_states[{i}]: {rollout_states[i]}")
                new_vector = [state[1] for state in rollout_states]
                #print(f"new_vector: {new_vector}, type: {type(new_vector)}")
                return values, np.array(new_vector)

    def backpropagate(self, value):
        self.value_sum += reward_to_norm(value)
        self.u_value_sum += value
        self.visit_count += 1

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
        
    def search(self, state):
        root = Node(self.game, self.args, state)
        
        
        for search in trange(self.args['num_searches']):
            node = root
            
            while len(node.children) > 4:
                node = node.select()
            #print(f"node.state: {node.state}, type: {type(node.state)}")
            values, is_terminal = self.game.get_value_and_terminated(node.state,1)
            #print("yes")
            if not is_terminal:
                node = node.expand()

            values, res = node.simulate(search)
            
            for value in values:
                with open(save_dir / "search_log.txt", "a") as f:
                    f.write(f"Search {search}: value = {value}\n")
                    #np.save(save_dir / f"res_{value}_in_sim{search}.npy", res)
                
                node.backpropagate(value)

        
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
    'num_searches': 5000
}

mcts = MCTS(env, args)

state = env.get_initial_state()

while True:
    #print(f"state1:{state}")
    mcts_probs, values = mcts.search(state)
    action = np.argmax(mcts_probs)
    print(f"state={state}, mcts_probs={mcts_probs}, values={values}, action={action}")    
    state = env.get_next_state(state, action)
    
    value, is_terminal = env.get_value_and_terminated(state,simulator_id)
    
    if is_terminal:
        print(value)
        break