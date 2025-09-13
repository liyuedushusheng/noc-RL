import numpy as np
from BookSim_wrapper import BookSimLatencySimulator

class NOC_env:
    def __init__(self, simulator):
        self.simulator = simulator
        self.row_count = simulator.NoC_height
        self.column_count = simulator.NoC_width
        self.action_size = self.row_count * self.column_count
        self.total_positions = self.row_count * self.column_count
        self.array_nums = simulator.get_array_num()
        self.num_layers = len(self.array_nums)
        self.final_reward = None
        self.final_latency_1 = None   # 存储 ifm_number=1 时的延迟
        self.final_latency_100 = None # 存储 ifm_number=100 时的延迟

        self.reset()

    def reset(self):
        return self.get_initial_state()

    def get_initial_state(self):
        position = []
        encoded_state = np.ones((self.row_count, self.column_count)) * -1
        return encoded_state, position

    def _convert_to_coordinates(self, pos):
        x = pos // self.column_count
        y = pos % self.column_count
        return (x, y)

    def get_value_and_terminated(self, states, parallel_nums):
        if parallel_nums==1:  # 处理单个状态
            states = [states]
            #print("yes")
            
        rewards = []
        location_infos=[]
        for state in states:
            #print(f"state:{state}")
            encoded_state, position = state
            terminated = len(position) == np.sum(self.array_nums)
            location_info = [[] for _ in range(self.num_layers)]
            if not terminated:
                rewards=[0 for _ in range(parallel_nums)]
                return rewards,False

            
            p = 0
            #print(f"location_info_len:{len(location_info)}")
            for layer in range(len(self.array_nums)):
                for _ in range(self.array_nums[layer]):
                    coors = self._convert_to_coordinates(position[p])
                    location_info[layer].append(coors)
                    p += 1
            #print("yes")
            location_infos.append(location_info)
        
        #print(f"location_infos_len:{len(location_infos)}")
        self.simulator.set_location_info(location_infos, parallel_nums)

        # 计算 ifm_number=50 时的延迟
        self.simulator.set_ifm_number(50, parallel_nums)
        latency_100 = self.simulator.InferenceTime(parallel_nums)

        # 计算延迟差作为奖励
        latency_diff = latency_100.copy()
        rewards = np.where(latency_100 >= 0, -latency_diff, -100000)
        
        if len(rewards) == 1:
            return rewards[0], terminated
        return np.array(rewards), terminated

    def get_valid_moves(self, state):
        encoded_state, position = state
        return (encoded_state.reshape(-1) == -1).astype(np.uint8)

    def get_next_state(self, state, action):
        encoded_state, position = state
        row = action // self.column_count
        column = action % self.column_count
        encoded_state[row, column] = len(position)
        position.append(action)
        return encoded_state, position

if __name__ == "__main__":
    simulator = BookSimLatencySimulator(
        config_addr="config_VGG.json",
        simulator_id=1,
        simulator_path="ISAAC_NOPU_MESH_FINAL_1"
    )
    env = NOC_env(simulator)
    state = env.get_initial_state()
    for i in range(25):
        action = i
        print(f"action: {action}")
        state = env.get_next_state(state, action)
        print(f"state: {state}")
        reward, terminated = env.get_value_and_terminated(state, 0)
        print(reward, terminated)