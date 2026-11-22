import pytest

from specforge.data.mmflash_preprocessing import (
    assistant_loss_mask,
    chat_messages,
    encode_record,
)


def test_image_in_first_question_is_removed_only_from_blind_prompt():
    turns = [
        {"from": "human", "value": "<image>\nWhat color?"},
        {"from": "gpt", "value": " Red."},
    ]
    visible, blind = chat_messages(turns, has_image=True)
    assert visible[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "What color?"},
    ]
    assert blind[0]["content"] == "What color?"
    assert visible[1] == blind[1] == {"role": "assistant", "content": " Red."}
    assert chat_messages(
        [{"role": "user", "content": "What color?"}, turns[1]], has_image=True
    ) == (visible, blind)
    with pytest.raises(ValueError, match="placeholder"):
        chat_messages(turns, has_image=False)


def test_supervision_starts_after_full_header_and_includes_eot():
    ids = [9, 10, 11, 2, 3, 8, 4, 10, 11, 5, 8, 4]
    assert assistant_loss_mask(ids, [10, 11], 8, last_turn_only=False) == [
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
    ]

    assert assistant_loss_mask(ids, [10, 11], 8, last_turn_only=True) == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
    ]


def test_scorer_rejects_unsupported_reply_formats_before_rendering():
    user = {"role": "user", "content": "What color?"}
    reply = {"role": "assistant", "content": "Red."}
    options = dict(
        processor=None, delimiters=(), last_turn_only=False, source="fixture"
    )
    with pytest.raises(ValueError, match="one user and one assistant"):
        encode_record({"conversations": [user, reply, user, reply]}, **options)
    with pytest.raises(ValueError, match="thinking off"):
        encode_record(
            {
                "conversations": [
                    user,
                    {**reply, "content": "<think>Reason.</think>Red."},
                ]
            },
            **options,
        )
