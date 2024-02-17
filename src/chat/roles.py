from dataclasses import dataclass
from enum import Enum


class ChatRole(Enum):
    """Enumeration representing the roles in a chat."""

    SYSTEM = 0
    ASSISTANT = 1
    HUMAN = 2


@dataclass
class RoleTokens:
    """Dataclass containing the tokens a model uses
    for each role in a chat."""

    system: str
    human: str
    assistant: str
