import flwr as fl
from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
from server import BCFLStrategy
import argparse
import os
import json
from web3 import Web3
import logging

DEFAULT_URL = "http://127.0.0.1:7545"
DEFAULT_ADDR = "0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"
DEFAULT_PATH = "ipfs_models/initial_model.pth"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Server - %(message)s')

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
    current_round = blockchain_utils.contract.functions.getCurrentRound().call()
    if current_round > round_num:
        print(f"当前轮次已是 {current_round}，无需推进到 {round_num + 1}")
        return
    tx = blockchain_utils.contract.functions.submitGlobalModel(round_num, cid).transact({'from': accounts[0]})
    blockchain_utils.web3.eth.wait_for_transaction_receipt(tx)
    print(f"已推进到轮次 {round_num + 1}")

def select_trainers_for_round(blockchain_utils, round_num, trainer_addresses):
    accounts = blockchain_utils.web3.eth.accounts
    current_round = blockchain_utils.contract.functions.getCurrentRound().call()
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

    from model import CNN
    abi = load_abi()
    run_server(args.url, args.addr, abi, args.rounds, args.clients, model_class=CNN)

if __name__ == "__main__":
    main()