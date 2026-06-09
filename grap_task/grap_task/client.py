import socket
import json
import struct
import time


class ChassisTCPClient:
    def __init__(self, host='172.31.4.222', port=8080):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.sock.connect((self.host, self.port))
        print(f"已连接到底盘控制器: {self.host}:{self.port}")

    def send_simple_message(self, msg_id=123, target_address=None, command="ok"):
        """
        发送简化的报文，只包含消息ID、目标地址和 command
        """
        if msg_id is None:
            msg_id = int(time.time() * 1000) % 10000  # 默认使用时间戳作为消息ID

        if target_address is None:
            target_address = "default_destination"  # 默认目标地址

        message = {
            "header": {
                "msg_id": msg_id,
                "timestamp": int(time.time() * 1000),
            },
            "target_address": target_address,
            "command": command,
        }

        json_str = json.dumps(message)
        data = json_str.encode('utf-8')

        # 添加长度前缀
        length_prefix = struct.pack('>I', len(data))
        self.sock.sendall(length_prefix + data)
        print(f"发送简化报文: ID={msg_id}, 地址={target_address}, command={command}")

        return msg_id

    def _recv_n_bytes(self, n: int) -> bytes:
        """
        从 socket 中精确读取 n 个字节
        """
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("连接已关闭或读取失败")
            data += chunk
        return data

    def recv_response(self):
        """
        接收一条带长度前缀的 JSON 响应
        """
        # 先读 4 字节长度
        raw_len = self._recv_n_bytes(4)
        (msg_len,) = struct.unpack('>I', raw_len)

        # 再读指定长度的数据
        raw_msg = self._recv_n_bytes(msg_len)
        json_str = raw_msg.decode('utf-8')
        resp = json.loads(json_str)
        print(f"收到响应: {resp}")
        return resp

    def close(self):
        self.sock.close()
        print("连接已关闭")


if __name__ == "__main__":
    client = ChassisTCPClient()
    try:
        client.connect()

        # 一直保持连接，循环发送/接收
        # 可以根据需要改成 while True + 键盘输入等
        
        client.send_simple_message(msg_id=123, target_address="motor_control", command="收银台")
        # client.send_simple_message(msg_id=123, target_address="motor_control", command="货架")


    except Exception as e:
        print(f"通信错误: {e}")
    finally:
        client.close()
