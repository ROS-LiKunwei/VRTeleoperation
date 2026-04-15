from abc import ABC, abstractmethod

from beavr.teleop.components import Component


class InputDevice(Component, ABC):
    @abstractmethod
    def stream(self):
        pass

    @abstractmethod
    def data_frequency(self):
        pass

    @abstractmethod
    def receive_data(self):
        pass

    @abstractmethod
    def process_data(self):
        pass
