import flwr as fl
import torch
import numpy as np
from model import load_model, save_model
from data import load_data
import logging
import io

class BCFLClient(fl.client.NumPyClient):
    def __init__(self, blockchain_utils, ipfs_utils, cid, model_class):
        self.blockchain_utils = blockchain_utils
        self.ipfs_utils = ipfs_utils
        self.cid = cid
        self.model = model_class()
        self.trainloader, self.testloader = load_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"客户端初始化完成，CID={self.cid}")

    def get_parameters(self, config):
        logging.info("获取参数")
        return []

    def fit(self, parameters, config):
        server_round = config.get("server_round", 1)
        logging.info(f"开始第 {server_round} 轮训练")

        round_num = self.blockchain_utils.get_current_round()
        logging.info(f"当前轮次: {round_num}")

        # 对于第一轮，使用 rounds[0].globalModelCID
        if round_num == 1:
            cid = self.blockchain_utils.get_global_model_cid(0)  # 获取初始模型
            logging.info(f"第一轮使用初始模型，CID={cid}")
        else:
            cid = self.blockchain_utils.get_global_model_cid(round_num - 1)  # 使用上一轮的全局模型
            logging.info(f"使用轮次 {round_num - 1} 的全局模型，CID={cid}")

        if not cid or cid == "":
            logging.error(f"轮次 {round_num} 无有效全局模型CID")
            return [np.array([], dtype=np.uint8)], 0, {"error": "无效CID"}

        try:
            model_bytes = self.ipfs_utils.download_model(cid)
            if not model_bytes:
                raise ValueError("下载模型失败，返回空字节")
            buffer = io.BytesIO(model_bytes)
            state_dict = torch.load(buffer, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logging.info(f"成功加载全局模型，CID={cid}")
        except Exception as e:
            logging.error(f"从IPFS下载模型失败: {e}")
            return [np.array([], dtype=np.uint8)], 0, {"error": str(e)}

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        for epoch in range(2):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            logging.info(f"完成第 {epoch + 1} 次epoch")

        new_cid = self.ipfs_utils.upload_model(self.model)
        if not new_cid:
            logging.error("上传更新模型到IPFS失败")
            return [np.array([], dtype=np.uint8)], 0, {"error": "上传失败"}
        logging.info(f"成功上传更新模型，CID={new_cid}")

        tx_receipt = self.blockchain_utils.submit_update_cid(round_num, new_cid)
        if not tx_receipt:
            logging.error("提交更新CID到区块链失败")
            return [np.array([], dtype=np.uint8)], 0, {"error": "提交CID失败"}
        logging.info(f"成功提交更新CID，交易哈希: {tx_receipt.transactionHash.hex()}")

        return [np.frombuffer(new_cid.encode('utf-8'), dtype=np.uint8)], len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        logging.info("评估模型")
        return 0.0, 0, {}