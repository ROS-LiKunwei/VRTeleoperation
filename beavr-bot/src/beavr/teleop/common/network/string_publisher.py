"""面向 Unity/PICO 的 ZMQ 字符串发布器。

消息使用 PUB/SUB 模式发送，由两帧组成：第一帧是订阅主题，第二帧是
UTF-8 字符串负载。目前 FA 标定流程用它向 PICO 发送语音资源键名。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import zmq

from beavr.teleop.common.network.utils import get_global_context


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PublisherKey:
    """标识一个可复用的发布端点，用作 manager 缓存键。"""

    host: str
    port: int


class ZMQStringPublisher:
    """发送 ``topic + UTF-8 payload`` 两帧消息的轻量 PUB socket。"""

    def __init__(self, host: str, port: int, context: zmq.Context | None = None):
        self.host = host
        self.port = int(port)
        # 全进程共享一个 ZMQ Context，避免每个语音发布器创建独立 I/O 线程。
        self._context = context or get_global_context()
        self._socket = self._context.socket(zmq.PUB)
        # 语音提示只关心较新的消息。队列积压超过 5 条时，不继续无限缓存。
        self._socket.setsockopt(zmq.SNDHWM, 5)
        # 进程退出时立即关闭，不等待尚未发送的提示，防止退出流程卡住。
        self._socket.setsockopt(zmq.LINGER, 0)
        # PICO 从远端主动 connect，因此服务端监听本机所有网络接口。
        self._socket.bind(f"tcp://*:{self.port}")
        # 同一个 socket 可能被多个 operator 线程调用。锁保证一条 multipart
        # 消息的 topic 和 payload 不会与另一条消息交叉。
        self._lock = threading.Lock()

    def publish(self, topic: str, payload: str) -> None:
        """非阻塞发送一条两帧字符串消息。

        ``zmq.NOBLOCK`` 保证网络或队列异常时不会阻塞遥操作控制线程；如果
        发送队列已满，pyzmq 会向调用方抛出 ``zmq.Again``。
        """

        with self._lock:
            self._socket.send_multipart(
                [topic.encode("utf-8"), payload.encode("utf-8")],
                flags=zmq.NOBLOCK,
            )

    def close(self) -> None:
        """立即关闭 socket，不等待发送队列排空。"""
        self._socket.close(linger=0)


class ZMQStringPublisherManager:
    """按网络端点复用字符串发布器的进程级单例。"""

    _instance: "ZMQStringPublisherManager | None" = None
    _lock = threading.Lock()

    def __init__(self, context: zmq.Context | None = None):
        self._context = context or get_global_context()
        self._publishers: dict[_PublisherKey, ZMQStringPublisher] = {}

    @classmethod
    def get_instance(cls, context: zmq.Context | None = None) -> "ZMQStringPublisherManager":
        """线程安全地获取单例；首次创建时采用传入的 Context。"""

        # 双重检查避免单例创建完成后，每次调用仍进入类级锁。
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(context)
        return cls._instance

    def publish(self, host: str, port: int, topic: str, payload: str) -> None:
        """获取或创建目标端点的 PUB socket，然后发送字符串消息。"""

        key = _PublisherKey(host=host, port=int(port))
        with self._lock:
            publisher = self._publishers.get(key)
            if publisher is None:
                publisher = ZMQStringPublisher(host, port, self._context)
                self._publishers[key] = publisher
                logger.info("Started string publisher topic=%s endpoint=tcp://*:%d", topic, port)
                # PUB/SUB 建连是异步的。首次 bind 后短暂等待，降低第一条语音
                # 在订阅关系尚未建立时丢失的概率（ZMQ slow joiner 问题）。
                time.sleep(0.05)
        # 发布器内部已有 socket 级锁，发送时无需继续占用 manager 的全局锁。
        publisher.publish(topic, payload)

    def close_all(self) -> None:
        """关闭并清空所有缓存 socket，供统一 ZMQ 资源清理流程调用。"""

        with self._lock:
            for publisher in self._publishers.values():
                publisher.close()
            self._publishers.clear()
