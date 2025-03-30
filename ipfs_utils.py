import io
import ipfshttpclient
import torch
import logging
import os
import tempfile

class IPFSUtils:
    def __init__(self, ipfs_api="/ip4/127.0.0.1/tcp/5001/http"):
        self.ipfs_api = ipfs_api
        try:
            with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                logging.info("成功连接到IPFS节点")
        except Exception as e:
            logging.error(f"无法连接到IPFS节点: {e}")
            raise

    def upload_model(self, model_or_path, use_file=False):
        """上传模型到IPFS，支持字节流或文件路径"""
        try:
            if use_file or not isinstance(model_or_path, torch.nn.Module):
                # 使用文件路径
                if isinstance(model_or_path, str):
                    file_path = model_or_path
                else:
                    file_path = tempfile.mktemp(suffix=".pth")
                    torch.save(model_or_path.state_dict(), file_path)
                with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                    cid = ipfs.add(file_path)["Hash"]
                if isinstance(model_or_path, torch.nn.Module):
                    os.remove(file_path)  # 清理临时文件
            else:
                # 使用字节流
                buffer = io.BytesIO()
                torch.save(model_or_path.state_dict(), buffer)
                buffer.seek(0)
                with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                    cid = ipfs.add_bytes(buffer.read())
            
            if not cid:
                logging.error("上传成功但未返回有效CID")
                return None
            logging.info(f"上传模型到IPFS，CID={cid}")
            return cid
        except Exception as e:
            logging.error(f"上传模型到IPFS失败: {e}")
            return None

    def download_model(self, cid, output_path=None):
        """从IPFS下载模型，支持字节流或保存到文件"""
        if not cid or not isinstance(cid, str) or cid.strip() == "":
            logging.error(f"无效的CID: {cid}")
            raise ValueError("CID 不能为空或无效")
        try:
            with ipfshttpclient.connect(self.ipfs_api) as ipfs:
                model_bytes = ipfs.cat(cid, timeout=60)
                logging.info(f"从IPFS下载CID {cid}")
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(model_bytes)
                logging.info(f"模型保存到 {output_path}")
            return model_bytes
        except Exception as e:
            logging.error(f"从IPFS下载模型失败: {e}")
            raise