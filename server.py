import flwr as fl
from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
from model import load_model, save_model, CNN  # 导入 CNN 作为示例模型
import torch
import logging
import io
from typing import Optional, Tuple, Dict, Type
from flwr.common import Parameters, Scalar, NDArrays
from web3 import Web3
import argparse
import os
import json

DEFAULT_URL = "http://127.0.0.1:7545"
DEFAULT_ADDR = "0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"
DEFAULT_PATH = "ipfs_models/initial_model.pth"

class BCFLStrategy(fl.server.strategy.Strategy):
    def __init__(self, blockchain_utils, ipfs_utils, model_class: Type[torch.nn.Module]):
        super().__init__()
        self.blockchain_utils = blockchain_utils
        self.ipfs_utils = ipfs_utils
        self.model_class = model_class  # 必须传入模型类
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from data import load_data
        self.testloader = load_data()[1]

    def initialize_parameters(self, client_manager):
        return fl.common.ndarrays_to_parameters([])

    def configure_fit(self, server_round, parameters, client_manager):
        num_clients = self.blockchain_utils.contract.functions.task().call()[3]
        available_clients = client_manager.sample(
            num_clients=num_clients,
            min_num_clients=num_clients
        )
        config = {"server_round": server_round}
        fit_ins = fl.common.FitIns(parameters, config)
        trainer_addresses = [self.blockchain_utils.web3.eth.accounts[i+1] for i in range(num_clients)]
        self.blockchain_utils.contract.functions.selectTrainersForRound(server_round, trainer_addresses).transact({'from': self.blockchain_utils.account})
        return [(client, fit_ins) for client in available_clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            logging.warning("未收到客户端结果")
            return None, {}

        cids = []
        for client, fit_res in results:
            ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            if not ndarrays or len(ndarrays) == 0 or len(ndarrays[0]) == 0:
                logging.warning(f"客户端 {client} 返回空参数")
                continue
            cid = ndarrays[0].tobytes().decode('utf-8')
            if not cid:
                logging.warning(f"客户端 {client} 返回无效CID")
                continue
            cids.append(cid)

        models = []
        for i, cid in enumerate(cids):
            try:
                model_bytes = self.ipfs_utils.download_model(cid)
                buffer = io.BytesIO(model_bytes)
                state_dict = torch.load(buffer, map_location=self.device)
                model = self.model_class().to(self.device)
                model.load_state_dict(state_dict)
                models.append(model)
                logging.info(f"加载客户端模型，CID={cid}")
            except Exception as e:
                logging.error(f"下载或加载CID {cid} 的模型失败: {e}")
                continue

        if not models:
            logging.error("无有效模型可聚合")
            return None, {}

        global_model = self.model_class().to(self.device)
        with torch.no_grad():
            for param in global_model.parameters():
                param.data.zero_()
            for model in models:
                for global_param, client_param in zip(global_model.parameters(), model.parameters()):
                    global_param.data += client_param.data / len(models)

        new_cid = self.ipfs_utils.upload_model(global_model)
        if not new_cid:
            logging.error("上传全局模型到IPFS失败")
            return None, {}

        self.blockchain_utils.submit_global_model(server_round, new_cid)
        logging.info(f"上传全局模型，CID={new_cid}")

        return fl.common.ndarrays_to_parameters([new_cid.encode('utf-8')]), {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Check if parameters contain tensors
        if not parameters.tensors:
            logging.info(f"Server Round {server_round} - No parameters provided for evaluation")
            return None, {}

        # Convert Parameters to list of numpy.ndarray
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        if not ndarrays:
            logging.info(f"Server Round {server_round} - Empty ndarray list after conversion")
            return None, {}

        # Extract CID: convert ndarray to bytes, then decode to string
        cid_bytes = ndarrays[0].tobytes()  # Convert numpy.ndarray to bytes
        cid = cid_bytes.decode('utf-8')    # Decode bytes to string

        try:
            # Download the model from IPFS using the CID
            self.ipfs_utils.download_model(cid, "global_model_eval.pth")
            # Load the model
            model = load_model("global_model_eval.pth", model_class=self.model_class, map_location=self.device).to(self.device)
            model.eval()

            # Define loss function
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            correct = 0
            total = 0

            # Evaluate the model
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    total_loss += criterion(outputs, target).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            # Calculate average loss and accuracy
            avg_loss = total_loss / len(self.testloader)
            accuracy = correct / total

            # Log results
            logging.info(f"Server Round {server_round} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            return avg_loss, {"accuracy": accuracy}

        except Exception as e:
            logging.error(f"Failed to evaluate global model: {e}")
            return None, {}

def load_abi(path="build/contracts/BCFL.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)["abi"]
    except FileNotFoundError:
        print(f"错误：找不到 {path} 文件，请先编译并部署合约。")
        exit(1)

def get_genesis_cid(path, ipfs_utils):
    if not os.path.exists(path):
        print(f"错误：初始模型文件 {path} 不存在")
        exit(1)
    cid = ipfs_utils.upload_model(path)
    if not cid:
        print("初始模型上传失败")
        exit(1)
    print(f"初始模型已上传至IPFS，CID={cid}")
    return cid

def is_task_initialized(blockchain_utils):
    return blockchain_utils.contract.functions.task().call()[4]

def initialize_task(blockchain_utils, cid, rounds, trainers):
    accounts = blockchain_utils.web3.eth.accounts
    if is_task_initialized(blockchain_utils):
        print("任务已初始化，无需重复操作")
        return
    tx = blockchain_utils.contract.functions.initialize(cid, rounds, trainers).transact({'from': accounts[0]})
    blockchain_utils.web3.eth.wait_for_transaction_receipt(tx)
    print(f"任务已初始化：CID={cid}, 总轮次={rounds}, 训练者数量={trainers}")

def advance_to_next_round(blockchain_utils, round_num, cid):
    accounts = blockchain_utils.web3.eth.accounts
    current_round = blockchain_utils.contract.functions.currentRound().call()
    if current_round > round_num:
        print(f"当前轮次已是 {current_round}，无需推进到 {round_num + 1}")
        return
    tx = blockchain_utils.contract.functions.submitGlobalModel(round_num, cid).transact({'from': accounts[0]})
    blockchain_utils.web3.eth.wait_for_transaction_receipt(tx)
    print(f"已推进到轮次 {round_num + 1}")

def select_trainers_for_round(blockchain_utils, round_num, trainer_addresses):
    accounts = blockchain_utils.web3.eth.accounts
    current_round = blockchain_utils.contract.functions.currentRound().call()
    if current_round != round_num:
        print(f"错误：当前轮次为 {current_round}，无法为轮次 {round_num} 选择训练者")
        return
    tx = blockchain_utils.contract.functions.selectTrainersForRound(round_num, trainer_addresses).transact({'from': accounts[0]})
    blockchain_utils.web3.eth.wait_for_transaction_receipt(tx)
    print(f"已为轮次 {round_num} 选择训练者")

def run_server(url, addr, abi, rounds, clients, model_class):
    blockchain_utils = BlockchainUtils(url, addr, abi)
    ipfs_utils = IPFSUtils()

    w3 = Web3(Web3.HTTPProvider(url))
    if not w3.is_connected():
        print(f"错误：无法连接到 {url}")
        exit(1)
    if w3.eth.get_code(addr).hex() == "0x":
        print(f"错误：在 {addr} 未找到已部署的合约")
        exit(1)

    genesis_cid = get_genesis_cid(DEFAULT_PATH, ipfs_utils)
    initialize_task(blockchain_utils, genesis_cid, rounds, clients)
    advance_to_next_round(blockchain_utils, 0, genesis_cid)

    trainer_addresses = w3.eth.accounts[1:clients+1]
    select_trainers_for_round(blockchain_utils, 1, trainer_addresses)

    strategy = BCFLStrategy(blockchain_utils, ipfs_utils, model_class=model_class)
    fl.server.start_server(
        server_address="localhost:8081",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy
    )

def main():
    parser = argparse.ArgumentParser(description="区块链联邦学习服务器")
    parser.add_argument("--clients", type=int, default=2, help="客户端数量")
    parser.add_argument("--rounds", type=int, default=3, help="训练轮次")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="区块链节点URL")
    parser.add_argument("--addr", type=str, default=DEFAULT_ADDR, help="智能合约地址")
    parser.add_argument("--path", type=str, default=DEFAULT_PATH, help="初始模型路径")
    args = parser.parse_args()

    from model import CNN  # 在 main 中导入默认模型
    abi = load_abi()
    run_server(args.url, args.addr, abi, args.rounds, args.clients, model_class=CNN)  # 默认使用 CNN

if __name__ == "__main__":
    main()