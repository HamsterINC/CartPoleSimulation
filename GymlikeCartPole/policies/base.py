from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def observe(self, state, action, reward, next_state, next_action, done):
        pass

    @abstractmethod
    def end_episode(self):
        pass

    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass