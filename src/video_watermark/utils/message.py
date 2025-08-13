from __future__ import annotations

from typing import Optional

import torch

from ..types import WatermarkBits


def random_bits(nbits: int, device: Optional[torch.device] = None) -> WatermarkBits:
    bits = torch.randint(0, 2, (1, nbits), device=device)
    return WatermarkBits(bits=bits.long())


def to_bitstring(bits: WatermarkBits) -> str:
    arr = bits.bits.detach().cpu().view(-1).tolist()
    return "".join(str(int(x)) for x in arr)


def from_bitstring(s: str, device: Optional[torch.device] = None) -> WatermarkBits:
    arr = torch.tensor([int(ch) for ch in s.strip()], device=device).view(1, -1).long()
    return WatermarkBits(bits=arr)


def assert_len(bits: WatermarkBits, nbits: int) -> None:
    if bits.length != nbits:
        raise ValueError(f"WatermarkBits length mismatch: got {bits.length}, expected {nbits}")



def encode_string_to_bits(message: str, nbits: int, device: Optional[torch.device] = None) -> WatermarkBits:
    """Encode a UTF-8 string into fixed-length bits with a 16-bit length prefix (big-endian).

    Layout: [len:16 bits][payload:len*8 bits][zero padding to nbits]
    """
    if nbits < 16:
        raise ValueError("nbits must be at least 16 to store length header")
    payload = message.encode("utf-8")
    max_payload_bytes = (nbits - 16) // 8
    if len(payload) > max_payload_bytes:
        raise ValueError(f"message too long: {len(payload)} bytes > capacity {max_payload_bytes}")

    length = len(payload)
    # length header 16 bits, big-endian
    length_bits = [(length >> (15 - i)) & 1 for i in range(16)]

    data_bits_list = []
    for b in payload:
        for i in range(8):
            data_bits_list.append((b >> (7 - i)) & 1)

    all_bits = length_bits + data_bits_list
    # pad with zeros
    if len(all_bits) < nbits:
        all_bits.extend([0] * (nbits - len(all_bits)))
    elif len(all_bits) > nbits:
        # Should not happen due to capacity check
        all_bits = all_bits[:nbits]

    tensor = torch.tensor(all_bits, device=device, dtype=torch.long).view(1, -1)
    return WatermarkBits(bits=tensor)


def decode_bits_to_string(bits: WatermarkBits) -> str:
    """Decode bits produced by encode_string_to_bits back to UTF-8 string."""
    b = bits.bits.view(-1).tolist()
    if len(b) < 16:
        raise ValueError("bits too short to contain header")
    length = 0
    for i in range(16):
        length = (length << 1) | int(b[i])
    required = 16 + length * 8
    if len(b) < required:
        raise ValueError("bits do not contain full payload")
    payload = []
    idx = 16
    for _ in range(length):
        val = 0
        for i in range(8):
            val = (val << 1) | int(b[idx + i])
        payload.append(val)
        idx += 8
    return bytes(payload).decode("utf-8")

