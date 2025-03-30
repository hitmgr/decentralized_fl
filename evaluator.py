import torch
from model import load_model
from data import load_data
from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
import logging
import json

class Evaluator:
    """模型评估器，用于评估客户端提交的模型并提交分数"""
    def __init__(self, blockchain_utils, ipfs_utils, round_num):
        """初始化评估器"""
        self.blockchain_utils = blockchain_utils
        self.ipfs_utils = ipfs_utils
        self.round_num = round_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testloader = load_data()[1]  # 加载测试数据

    def evaluate_model(self, cid):
        """评估指定CID的模型，返回准确率"""
        try:
            model_path = f"eval_model_{cid}.pth"
            self.ipfs_utils.download_model(cid, model_path)
            model = load_model(model_path, map_location=self.device).to(self.device)
            model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.testloader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    total_loss += criterion(outputs, target).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            accuracy = correct / total
            logging.info(f"评估CID {cid} 的模型，准确率: {accuracy:.4f}")
            return accuracy
        except Exception as e:
            logging.error(f"评估CID {cid} 的模型失败: {e}")
            return 0.0

    def submit_scores(self):
        """为当前轮次的训练者提交分数"""
        trainers = self.blockchain_utils.get_selected_trainers(self.round_num)
        for trainer in trainers:
            # 获取训练者提交的更新CID
            cid = self.blockchain_utils.contract.functions.rounds(self.round_num).updates(trainer).call()
            if cid and cid != "":
                score = self.evaluate_model(cid)
                # 将准确率转换为0-100的整数分数
                tx = self.blockchain_utils.contract.functions.submitScore(
                    self.round_num, trainer, int(score * 100)
                ).transact({'from': self.blockchain_utils.account})
                self.blockchain_utils.web3.eth.wait_for_transaction_receipt(tx)
                logging.info(f"提交训练者 {trainer} 的分数: {score}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - Evaluator - %(message)s')

    def load_abi(path="build/contracts/BCFL.json"):
        with open(path, "r") as f:
            return json.load(f)["abi"]

    blockchain_utils = BlockchainUtils("http://127.0.0.1:7545", "0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab", load_abi())
    ipfs_utils = IPFSUtils()
    round_num = 1  # 示例轮次
    evaluator = Evaluator(blockchain_utils, ipfs_utils, round_num)
    evaluator.submit_scores()