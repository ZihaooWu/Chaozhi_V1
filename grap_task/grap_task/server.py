import socket
import json
import struct
import threading

class ChassisMessageParser:
    """
    用于解析来自 ChassisTCPClient 的报文：
    报文格式 = 4 字节大端长度前缀 + JSON 字符串（utf-8 编码）
    """
    def __init__(self, conn: socket.socket):
        self.conn = conn

    def _recv_n_bytes(self, n: int) -> bytes:
        data = b''
        while len(data) < n:
            chunk = self.conn.recv(n - len(data))
            if not chunk:
                # 这里抛 ConnectionError，外层可以据此判断客户端已断开
                raise ConnectionError("连接关闭或无法继续读取数据")
            data += chunk
        return data

    def recv_message(self) -> dict:
        # 1. 读取长度前缀
        raw_len = self._recv_n_bytes(4)
        (msg_len,) = struct.unpack('>I', raw_len)

        # 2. 读取 msg_len 字节的 JSON 内容
        raw_msg = self._recv_n_bytes(msg_len)

        # 3. 解析 JSON
        json_str = raw_msg.decode('utf-8')
        message = json.loads(json_str)
        return message

    def recv_command(self):
        message = self.recv_message()
        return message.get("command")
    
    def send_response(self, resp_dict: dict):
        """将响应 dict 发送给客户端（带 4 字节长度前缀）"""
        try:
            data = json.dumps(resp_dict).encode('utf-8')
            length_prefix = struct.pack('>I', len(data))
            self.conn.sendall(length_prefix + data)
            print(f"发送响应给 {self.conn.getpeername()}: {resp_dict}")
            return True
        except OSError:
            return False


class ChassisTCPServer:
    """
    基于 socket 的简单 TCP 服务端
    自动接收并解析客户端消息，客户端断开后继续监听新客户端
    """
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd = None
        self._running = False
        self.parser = None

    def start(self):
        """启动服务器（阻塞版，一直跑）"""
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(5)  # backlog 可以设大一点
        self._running = True
        print(f"服务器已启动，监听 {self.host}:{self.port}")
        try:
            while self._running:
                print("等待客户端连接...")
                conn, addr = self.server_sock.accept()
                print(f"客户端已连接：{addr}")

                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True
                ).start()

        finally:
            self.server_sock.close()
            print("服务器关闭")
        
    def _handle_client(self, conn, addr):
        self.parser = ChassisMessageParser(conn)
        try:
            while True:
                message = self.parser.recv_message()
                if not message:
                    break

                print(f"[{addr}] 收到消息：{message}")
                cmd = message.get("command", "")
                if cmd:
                    self.cmd = cmd
                    print(f"[{addr}] 更新 command: {self.cmd}")

        except ConnectionError as e:
            print(f"客户端 {addr} 已断开：{e}")
        except Exception as e:
            print(f"处理客户端 {addr} 时发生异常：{e}")
        finally:
            conn.close()
            print(f"与客户端 {addr} 的连接已关闭")


if __name__ == "__main__":
    server = ChassisTCPServer(host="0.0.0.0", port=9090)
    server.start()
