"""Host-side sizing policies for sequence padding and block anchor sampling."""


def padded_sequence_length(length: int) -> int:
    """Use the longest sequence in the batch by default."""
    return length


def anchor_capacity(valid) -> int:
    """Use the largest number of adjacent valid token pairs in a row."""
    return int((valid[:, :-1] & valid[:, 1:]).sum(-1).max())
