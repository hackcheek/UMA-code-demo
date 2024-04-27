from enum import Enum


class State(str, Enum):
    RUNNING = 'running'
    STOPPED = 'stopped'
    PENDING = 'pending'
    COMPLETED = 'completed'
