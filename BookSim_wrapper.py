import math
import os
import subprocess
import json
from write_booksim_config import write_booksim_config
import numpy as np
import threading


class BookSimLatencySimulator:

    def __init__(self, config_addr, simulator_id=0, simulator_path=None):
        """
        Read from a configuration json file and initialize the simulator.

        :param config_addr: The address of the configuration json file.
        """

        # Check if the configuration file exists and is a json file.
        assert config_addr.endswith('.json'), "The configuration file must be a json file."
        self.config_addr = config_addr
        self.simulator_path = simulator_path
        with open(config_addr, 'r') as file:
            config = json.load(file)

            self.num_of_layer = config["number of layers"]
            self.num_of_repeats = config["number of repeats"]
            self.array_num = config["number of nodes"]

            self.NoC_width = config["NoC Width"]
            self.NoC_height = config["NoC Height"]

            self.location_info = [[(-1, -1) for _ in range(self.array_num[i])] for i in range(self.num_of_layer)]

            assert sum(self.array_num) <= self.NoC_width * self.NoC_height, "The number of nodes exceeds the NoC size."

            self.ifm_number = 1  # 添加默认值
        
        self.simulator_id = simulator_id
    
    def _convert_to_coordinates(self, pos):
        x = pos % self.NoC_width
        y = pos // self.NoC_width
        return (x, y)

    def get_array_num(self) -> list:
        """
        Instead of calculating the number of arrays for each layer, for BookSim the number of arrays is given in the config file.

        :return: a list of the number of arrays for each layer
        """
        return self.array_num

    def set_location_info(self, location_infos, parallel_nums):
        """
        Set the location information of the nodes in the network.

        :param location_info: Detailed information of all L layers' locations, should be a list of lists, each list
        denotes a layer's coordinates.

        :return: None
        """
        assert len(location_infos) == parallel_nums, "location_infos 的长度必须等于 parallel_nums"

        for id in range(parallel_nums):
            location_info = location_infos[id]  # 当前模拟的位置信息
            # 重新编码棋盘上的点
            mapping = {}
            for layer_idx, layer_nodes in enumerate(location_info):
                layer_name = f"layer{layer_idx + 1} nodes"
                mapping[layer_name] = []
                for node in layer_nodes:
                    if node == (-1, -1):
                        # Unplaced cell
                        del mapping[layer_name] # HJ
                        break # HJ
                        # continue
                    x, y = node
                    assert 0 <= x < self.NoC_width and 0 <= y < self.NoC_height, f"The node at ({x}, {y}) is out of the NoC range."
                    # 计算重新编码后的索引
                    encoded_index = x + y * self.NoC_width
                    encoded_index = int(encoded_index)
                    mapping[layer_name].append(encoded_index)

            # dir_file_path = os.path.join(os.getcwd(), f"ISAAC_NOPU_MESH_FINAL_{self.simulator_id}", "src", "mapping_vgg8.json")
            #dir_file_path = os.path.join(self.simulator_path, "src", f"mapping_vgg8_{id}.json")
        
            dir_file_path = os.path.join(os.getcwd(), "ISAAC_NOPU_MESH_FINAL_1", "src", f"mapping_vgg8_{id}.json")
            # print(f"dir_file_path={dir_file_path}")
            # Write dictionary to JSON file, each layer in a new line
        
            with open(dir_file_path, "w") as file:
                file.write(json.dumps(mapping))  #
      
                    #json.dump(mapping, file)
                    #file.flush()
            # 更新当前对象的属性
            self.location_info = location_info

            # Write new simulator configuration into the configuration file
            write_booksim_config(self.config_addr, id)
            # write_booksim_config(self.config_addr)

        return None

    def set_ifm_number(self, ifm_number,parallel_nums):
        """
        设置 ifm_number 的值并更新配置文件
        
        :param ifm_number: 新的 ifm_number 值
        """
        self.ifm_number = ifm_number
        for id in range(parallel_nums):
            # 更新配置文件中的 ifm_number
            with open(os.path.join(self.simulator_path, "src", "examples", f"mesh88_lat_{id}"), 'r') as file:
            #with open(os.path.join(self.simulator_path, "src", "examples", f"mesh88_lat_{id}"), 'r') as file:
                lines = file.readlines()
            
            for i, line in enumerate(lines):
                if line.strip().startswith("ifm_number"):
                    lines[i] = f"ifm_number = {self.ifm_number};\n"
                    break
                
            # with open(os.path.join(os.getcwd(), "ISAAC_NOPU_MESH_FINAL", "src", "examples", "mesh88_lat"), 'w') as file:
            with open(os.path.join(self.simulator_path, "src", "examples", f"mesh88_lat_{id}"), 'w') as file:
                file.writelines(lines)

    def InferenceTime(self, parallel_nums):
        if all(all(item == (-1, -1) for item in sublist) for sublist in self.location_info[1:]):
            return 0
            
        booksim_path = os.path.join(self.simulator_path, "src", "booksim")
       
        subprocess.run([booksim_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #file_path = os.path.join(self.simulator_path, "src", "output_file", f"watch_file_{simulator_id}")
        numbers=[]
        for id in range(parallel_nums):
            file_path = os.path.join(self.simulator_path, "src", "output_file", f"watch_file_{id}")
            # 初始化 number 变量
            number = None
            with open(file_path, 'r') as file:
                for line in file:
                    number = int(line.split()[0])
        
            # 检查 number 是否被正确赋值
            if number is not None:
                numbers.append(number)
            else:
                print("文件为空或没有内容，无法读取 number。")
                return None
        numbers=np.array(numbers)
        return numbers


def convert_to_coordinates(pos, NoC_width, NoC_height):
    x = pos // NoC_width
    y = pos % NoC_width
    return (x, y)

# 测试代码
if __name__ == "__main__":
    # 创建 LatencySimulator 实例
    simulator_id = 1
    simulator = BookSimLatencySimulator(
        config_addr="./config_VGG.json",
        simulator_id=simulator_id,
        simulator_path=f"./ISAAC_NOPU_MESH_FINAL_{simulator_id}",
    )
    # print(f"InferenceTime returned: {result}")

    """
    0  1  2  3  4
    5  6  7  8  9
    10 11 12 13 14
    15 16 17 18 19
    20 21 22 23 24
    """

    # 调用 set_location_info 方法
    # new_location_info = [
    #     [(4, 4)],  # Layer 1
    #     [(2, 0)],  # Layer 2
    #     [(3, 0), (1, 0)],  # Layer 3
    #     [(4, 0), (0, 1), (1, 1)],  # Layer 4
    #     [(3, 2), (3, 1), (4, 1), (0, 2), (1, 2), (2, 2)],  # Layer 5
    #     [(2, 1), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3)],  # Layer 6
    #     [(4, 3), (0, 4), (1, 4), (2, 4), (3, 4), (0, 0)],  # Layer 7
    # ]
    # new_location_info = [   
    #     [(3, 2)],  # Layer 1
    #     [(3, 3)],  # Layer 2
    #     [(4, 3), (1, 4)],  # Layer 3
    #     [(3, 4), (2, 4), (4, 0)],  # Layer 4
    #     [(0, 3), (1, 2), (2, 0), (4, 4), (1, 1), (1, 0)],  # Layer 5
    #     [(1, 3), (3, 1), (0, 4), (0, 2), (4, 1), (0, 0)],  # Layer 6
    #     [(2, 3), (4, 2), (2, 2), (2, 1), (0, 1), (3, 0)],  # Layer 7
    # ]
    # new_location_info = [   
    #     [(3, 2)],  # Layer 1
    #     [(3, 3)],  # Layer 2
    #     [(4, 3), (1, 4)],  # Layer 3
    #     [(3, 4), (2, 4), (4, 0)],  # Layer 4
    #     [(0, 3), (1, 2), (2, 0), (4, 4), (1, 1), (1, 0)],  # Layer 5
    #     [(1, 3), (3, 1), (0, 4), (0, 2), (4, 1), (0, 0)],  # Layer 6
    #     [(2, 3), (4, 2), (2, 2), (2, 1), (0, 1), (3, 0)],  # Layer 7
    # ]
    # new_location_info = [   
    #     [(3, 2)],  # Layer 1
    #     [(2, 2)],  # Layer 2
    #     [(2, 3), (3, 3)],  # Layer 3
    #     [(2, 1), (1, 2), (2, 4)],  # Layer 4
    #     [(4, 4), (2, 0), (1, 4), (0, 2), (4, 2), (3, 4)],  # Layer 5
    #     [(0, 0), (0, 3), (3, 1), (1, 3), (1, 1), (4, 1)],  # Layer 6
    #     [(4, 3), (0, 1), (4, 0), (3, 0), (1, 0), (0, 4)],  # Layer 7
    # ] # latency=1357

    # new_location_info = [   
    #     [(2, 2)],  # Layer 1
    #     [(3, 2)],  # Layer 2
    #     [(2, 3), (3, 3)],  # Layer 3
    #     [(2, 1), (1, 2), (2, 4)],  # Layer 4
    #     [(4, 4), (2, 0), (1, 4), (0, 2), (4, 2), (3, 4)],  # Layer 5
    #     [(0, 0), (0, 3), (3, 1), (1, 3), (1, 1), (4, 1)],  # Layer 6
    #     [(4, 3), (0, 1), (4, 0), (3, 0), (1, 0), (0, 4)],  # Layer 7
    # ] # latency=1359
    
    # new_location_info = [   
    #     [(3, 2)],  # Layer 1
    #     [(2, 2)],  # Layer 2
    #     [(2, 3), (3, 3)],  # Layer 3
    #     [(2, 1), (1, 2), (2, 4)],  # Layer 4
    #     [(4, 4), (2, 0), (1, 4), (0, 2), (4, 2), (3, 4)],  # Layer 5
    #     [(0, 0), (0, 3), (3, 1), (1, 3), (1, 1), (4, 1)],  # Layer 6
    #     [(4, 3), (0, 1), (4, 0), (3, 0), (1, 0), (0, 4)],  # Layer 7
    # ] # latency=1359

    num_layers = simulator.get_array_num()
    # position = np.arange(np.sum(num_layers))
    # position = np.random.permutation(position)
    position = np.load("/home/xieziang/Desktop/workspace/10*10/MCTS_VGG16/C-First/mcts-vgg16_0/save_dir/vgg16_norm-0310/max_value_-33312.npy")
    print(f"position={position}")
    new_location_info = []
    for layers in num_layers:
        new_location_info.append([])
    p = 0
    for layer in range(len(num_layers)):   
        for _ in range(num_layers[layer]):
            coors = convert_to_coordinates(position[p], simulator.NoC_width, simulator.NoC_height)
            new_location_info[layer].append(coors)
            p += 1
    print(f"new_location_info={new_location_info}")
    simulator.set_location_info(new_location_info, 1)
    simulator.set_ifm_number(1, 1)
    latency_1 = simulator.InferenceTime(1)
    print(f"latency_1={latency_1}")
    simulator.set_ifm_number(100, 1)
    latency_100 = simulator.InferenceTime(1)
    print(f"latency_100={latency_100}")

    print(latency_100 - latency_1)