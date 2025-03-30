import flwr as fl
from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
from client import BCFLClient
import argparse
import logging
import json

DEFAULT_URL = "http://127.0.0.1:7545"
DEFAULT_ADDR = "0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - Client %(client_id)d - %(message)s')

def load_abi(path="build/contracts/BCFL.json"):
    try:
        with open(path, "r") as f:
            return json.load(f)["abi"]
    except FileNotFoundError:
        print(f"错误：找不到 {path} 文件，请先编译并部署合约。")
        exit(1)

def start_client(url, addr, abi, cid, account_idx, model_class):
    blockchain_utils = BlockchainUtils(url, addr, abi)
    account = blockchain_utils.web3.eth.accounts[account_idx]
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter(f'%(asctime)s - Client {cid} - %(message)s')
    )
    logging.info(f"启动客户端，使用账户 {account}")

    ipfs_utils = IPFSUtils()
    client = BCFLClient(blockchain_utils, ipfs_utils, cid, model_class=model_class)
    
    fl.client.start_client(
        server_address="localhost:8081",
        client=client.to_client(),
        grpc_max_message_length=fl.common.GRPC_MAX_MESSAGE_LENGTH,
        root_certificates=None,
        insecure=True
    )

def main():
    parser = argparse.ArgumentParser(description="区块链联邦学习客户端")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="区块链节点URL")
    parser.add_argument("--addr", type=str, default=DEFAULT_ADDR, help="智能合约地址")
    parser.add_argument("--cid", type=int, required=True, help="客户端ID（如 1, 2）")
    parser.add_argument("--account_idx", type=int, default=1, help="使用的账户索引（从1开始）")
    args = parser.parse_args()

    from model import CNN
    abi = load_abi()
    start_client(args.url, args.addr, abi, args.cid, args.account_idx, model_class=CNN)

if __name__ == "__main__":
    main()