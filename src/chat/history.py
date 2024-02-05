"""Chat history related components."""
from dataclasses import dataclass
from src.chat.roles import ChatRole

@dataclass
class HistoryMessage:
    """Dataclass representing a message in a chat history."""
    role: ChatRole
    content: str