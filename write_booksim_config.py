import json
import os
import shutil


DEBUG = False

MAPPING_PATH = os.path.join(os.getcwd(), "ISAAC_NOPU_MESH_FINAL_1", "src", "mapping_vgg8.json")
#RESULT_PATH = os.path.join(os.getcwd(), "ISAAC_NOPU_MESH_FINAL", "src", "examples", "mesh88_lat")
RESULT_PATH = os.path.join(os.getcwd(), "ISAAC_NOPU_MESH_FINAL_1", "src", "examples")
def write_booksim_config(config_path,id, mapping_path= None, result_path= RESULT_PATH):
    """
    Read from a configuration json file to get the basic information of the network,
    Read from a mapping json file to get the location information of the nodes,
    Write the configuration of the BookSim simulator to a file.

    :param config_path: The configuration json file path, containing the basic information of the network.
    :param mapping_path: The mapping json file path, containing the location information of the position of the nodes.
    :param result_path: Configuration file path of the BookSim simulator.

    :return: None
    """
    mapping_dir = os.path.join(os.getcwd(), "ISAAC_NOPU_MESH_FINAL_1", "src")
    mapping_path = os.path.join(mapping_dir, f"mapping_vgg8_{id}.json")

    original_path = os.path.join(result_path, "mesh88_lat")
    result_path = os.path.join(result_path, f"mesh88_lat_{id}")
    if (not os.path.exists(result_path)):
        shutil.copy2(original_path, result_path)
    """
    # 检查文件是否存在且非空
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    if os.path.getsize(mapping_path) == 0:
        raise ValueError(f"Mapping file is empty: {mapping_path}")
    
    try:
        with open(mapping_path, "r") as f:
            layer_mapping_data = json.load(f)
    except json.JSONDecodeError as e:
        # 打印文件内容用于调试
        with open(mapping_path, "r") as f:
            content = f.read()
        print(f"Error decoding JSON from {mapping_path}. Content:")
        print(content)
        raise
    """
    with open(result_path, 'r') as file:
        lines = file.readlines()

    # 从 cur_mapping.json文件读取映射数据
    
    with open(mapping_path, "r") as f:
        layer_mapping_data = json.load(f)

    with open(config_path, 'r') as file:
        config = json.load(file)

    # 将 "layerX nodes" 格式转换为简单的 "layerX" 作为键
    layer_ranges = {layer.replace(" nodes", ""): nodes for layer, nodes in layer_mapping_data.items()}

    # 定义发送规则
    send_rules = config["sending rules"]

    # 以数据包计数
    send_rules_packet = config["sending rules packet"]

    # 初始化节点层次列表，大小为最大节点数+1，初始值为0
    max_node = max([node for nodes in layer_mapping_data.values() for node in nodes])
    node_layer_list = [0] * (max_node + 1)


    # 计算每个 layer 对应的平均数据包数（取整）
    average_packets = {}

    for layer, targets in send_rules.items():
        total_packets = sum(packet_count for _, packet_count in targets)
        average_packets[layer] = int(total_packets / len(targets))

    # 初始化每个节点接收到的数据包计数，根据 mapping.json 中的最大节点编号
    all_nodes = {node for nodes in layer_ranges.values() for node in nodes}
    max_node_id = max(all_nodes)
    node_received_packets = {node: 0 for node in range(max_node_id + 1)}

    # 遍历发送规则并累加每个节点的接收数据包数
    for layer, targets in send_rules_packet.items():
        if layer in layer_ranges:
            sender_nodes = layer_ranges[layer]
            for target_layer, packets_per_node in targets:
                if target_layer in layer_ranges:
                    target_nodes = layer_ranges[target_layer]

                    # 每个目标节点接收来自发送层的每个节点的数据包
                    for node in target_nodes:
                        node_received_packets[node] += len(sender_nodes) * packets_per_node

    # 用于存储每个层发送的总 flit 数和每个层的平均 flit 大小
    layer_flit_counts = {}
    layer_avg_flit_sizes = {}

    # 遍历 send_rules，计算每个层发送的 flit 数量

    for layer, rules in send_rules.items():
        total_flits = 0
        num_target_nodes_total = 0  # 记录所有目标节点的总数量

        for target_layer, flits_per_node in rules:
            target_nodes = layer_ranges.get(target_layer, [])
            num_target_nodes = len(target_nodes)  # 目标节点的数量

            # 计算该层向目标层发送的 flit 数量
            total_flits += flits_per_node * num_target_nodes
            num_target_nodes_total += num_target_nodes  # 累加目标节点的总数量

        # 计算每层向各个目标节点发送的平均 flit 数量
        if num_target_nodes_total > 0:
            avg_flit_size = total_flits / num_target_nodes_total
        else:
            avg_flit_size = 0  # 如果没有目标节点，设为 0

        # 存储结果
        layer_flit_counts[layer] = total_flits
        layer_avg_flit_sizes[layer] = avg_flit_size

    # 提取所有层的平均 flit 大小
    average_flit_sizes = [int(layer_avg_flit_sizes[layer]) for layer in send_rules]

    # 输出格式化为 {24,40,36,36,36,24,16,16,16,16,16,16,16}
    formatted_output = "Pu_packet_size = {" + ",".join(map(str, average_flit_sizes)) + "};"
    if DEBUG: print(formatted_output)

    for i, line in enumerate(lines):
        if line.strip().startswith("Pu_packet_size"):

            lines[i] = formatted_output + '\n'
            break

    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)


    # 输出结果格式化为 {value1, value2, ...} 格式
    result = "Layer_packet_num = {" + ",".join(str(node_received_packets[node]) for node in sorted(node_received_packets.keys())) + "};"
    if DEBUG: print(result)

    for i, line in enumerate(lines):
        if line.strip().startswith("Layer_packet_num"):

            lines[i] = result + '\n'
            break

    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)

    # 将数据转换为 layer_nodes 格式
    layer_nodes = {layer.replace(" nodes", ""): len(nodes) for layer, nodes in layer_mapping_data.items()}
    # 用于记录每个层向多少个目标层的节点发送数据
    layer_send_count = {layer: 0 for layer in layer_nodes}

    # 遍历 send_rules_packet 中的每个层，统计其目标层节点数的总和
    for layer, rules in send_rules_packet.items():
        total_target_nodes = 0

        for target_layer, _ in rules:
            # 获取目标层的节点数并进行累加
            target_nodes_count = layer_nodes.get(target_layer, 0)
            total_target_nodes += target_nodes_count

        # 记录该层向目标层发送数据的节点数量总和
        layer_send_count[layer] = total_target_nodes
    formatted_output = "Pu_send_num = {" + ",".join(str(count) for count in layer_send_count.values()) + "};"

    if DEBUG: print(formatted_output)


    for i, line in enumerate(lines):
        if line.strip().startswith("Pu_send_num"):

            lines[i] = formatted_output + '\n'
            break

    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)

    # 填充节点层次信息
    for layer, nodes in layer_mapping_data.items():
        # 从 layer 名称中提取数字部分作为层编号
        layer_num = int(layer.split()[0].replace("layer", ""))
        for node in nodes:
            node_layer_list[node] = layer_num

    formatted_output = "Node_layer = {" + ",".join(map(str, node_layer_list)) + "};"
    if DEBUG: print(formatted_output)

    for i, line in enumerate(lines):
        if line.strip().startswith("Node_layer"):

            lines[i] = formatted_output + '\n'
            break

    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)

     # 用于记录每个层向目标层发送的节点
    layer_send_to_nodes = {layer: [] for layer in layer_ranges}

    # 遍历发送规则，输出每个层向哪些节点发送数据
    for layer, rules in send_rules.items():
        if layer not in layer_ranges:
            continue
        sender_nodes = layer_ranges[layer]
        
        for target_layer, _ in rules:
            if target_layer not in layer_ranges:
                continue
            
            target_nodes = layer_ranges[target_layer]

            
            # 记录该层向哪些节点发送数据
            layer_send_to_nodes[layer].extend(target_nodes)

    # 格式化输出为 {{1, 2}, {2, 3}, ...} 的形式
    formatted_output = "Dest_pu = {" + ",".join("{" + ",".join(map(str, nodes)) + ",}" for nodes in layer_send_to_nodes.values()) + "};"

    for i, line in enumerate(lines):
        if line.strip().startswith("Dest_pu"):

            lines[i] = formatted_output + '\n'
            if DEBUG: print(lines[i])
            break  
    
    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)

    # 初始化每个节点接收数据包数的累加器
    max_value = max(max(nodes) for nodes in layer_mapping_data.values())
    node_received_data = [0]*(max_value+1)
    node_received_count = [0]*(max_value+1)
    node_average_data = [0]*(max_value+1)

    # 遍历发送规则累加每个节点接收的数据包大小
    for sender_layer, rules in send_rules.items():
        if sender_layer not in layer_ranges:
            continue
        sender_nodes = layer_ranges[sender_layer]
        for target_layer, data_size in rules:
            if target_layer not in layer_ranges:
                continue
            target_nodes = layer_ranges[target_layer]

            
            # 每个目标节点接收到的数据包大小累加
            for node in target_nodes:
                node_received_data[node] += data_size
                node_received_count[node] += 1



    # 计算每个节点接收的数据的平均大小
    for node in range(max_value+1):
        if node_received_count[node] != 0:
            node_average_data[node] = node_received_data[node] // node_received_count[node]
        else:
            node_average_data[node] = 0  # 如果接收计数为0，设置平均值为0
    
    node_average_data_dict = {node: node_average_data[node] for node in range(max_value + 1)}
    formatted_output = "Input_size = {" + ",".join(str(avg) for avg in node_average_data_dict.values()) + "};"
    
    for i, line in enumerate(lines):
        if line.strip().startswith("Input_size"):

            lines[i] = formatted_output + '\n'
            break  

    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)


     # 获取最后一个 layer 的键
    last_layer_key = list(layer_mapping_data.keys())[-1]
    
    # 提取最后一个 layer 的序号
    last_layer = int(last_layer_key.split()[0].replace("layer", ""))
    
    # 获取最后一个 layer 的节点数量
    last_layer_nodes = layer_mapping_data[last_layer_key]
    last_layer_num = len(last_layer_nodes)
    
    # 格式化输出
    lastlayer_output = f"Lastlayer = {last_layer};"
    lastlayernum_output = f"Lastlayernum = {last_layer_num};"

    for i, line in enumerate(lines):
        if line.strip().startswith("Lastlayer"):

            lines[i] = lastlayer_output + '\n'
            break  
        
    for i, line in enumerate(lines):
        if line.strip().startswith("Lastlayernum"):

            lines[i] = lastlayernum_output + '\n'
            break  

    # 将修改后的内容写回文件
    with open(result_path, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    write_booksim_config("./config_VGG.json")
    