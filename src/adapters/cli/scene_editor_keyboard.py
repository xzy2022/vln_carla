from __future__ import annotations

import ctypes
from typing import Any, Protocol


VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_U = 0x55
VK_O = 0x4F
VK_C = 0x43
VK_B = 0x42
VK_ESC = 0x1B


class KeyboardInterface(Protocol):
    def is_pressed(self, vk_code: int) -> bool:
        ...


class WindowsKeyboard(KeyboardInterface):
    def __init__(self) -> None:
        try:
            self._user32: Any = ctypes.windll.user32
        except AttributeError as exc:
            raise RuntimeError("Windows keyboard backend requires ctypes.windll.user32.") from exc

    def is_pressed(self, vk_code: int) -> bool:
        return bool(self._user32.GetAsyncKeyState(vk_code) & 0x8000)
