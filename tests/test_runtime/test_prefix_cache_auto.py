from unittest import mock

from specforge.inference.prefix_cache import auto_prefix_cache_modes


def _prompts(rows):
    return [{"payload": {"input_ids": ids}} for ids in rows]


def _server(page_size=4, replicas=1):
    return {
        "internal_states": [
            {"spec_capture_prefix_page_size": page_size} for _ in range(replicas)
        ]
    }


@mock.patch("specforge.inference.prefix_cache._get_server_info")
def test_auto_uses_complete_pages_including_after_replica_split(server_info):
    server_info.return_value = _server(replicas=2)
    common = [[1, 2, 3, 4, token] for token in range(10, 26)]
    short = [[1, 2, 3, 4] for _ in common]
    random_rows = [[token, 2, 3, 4, 5] for token in range(10, 26)]
    assert auto_prefix_cache_modes(_prompts(common), ["http://a"], lease=2, seed=0) == [
        True
    ]
    for rows in (short, random_rows):
        assert auto_prefix_cache_modes(
            _prompts(rows), ["http://a"], lease=2, seed=0
        ) == [False]


@mock.patch("specforge.inference.prefix_cache._get_server_info")
def test_auto_falls_back_for_unavailable_or_incompatible_servers(server_info):
    server_info.side_effect = [
        _server(),
        {"internal_states": [{}]},
        OSError("not ready"),
        {
            "internal_states": [
                {"spec_capture_prefix_page_size": 4},
                {"spec_capture_prefix_page_size": 8},
            ]
        },
    ]
    prompts = _prompts([[1, 2, 3, 4, token] for token in range(40)])
    assert auto_prefix_cache_modes(
        prompts, ["http://a", "http://b", "http://c", "http://d"], lease=2, seed=0
    ) == [True, False, False, False]


@mock.patch("specforge.inference.prefix_cache._get_server_info")
def test_auto_reads_at_most_512_prepared_prompts(server_info):
    server_info.return_value = _server()
    prompts = mock.MagicMock()
    prompts.__len__.return_value = 10000
    prompts.__getitem__.return_value = {"payload": {"input_ids": [1, 2, 3, 4, 5]}}
    auto_prefix_cache_modes(prompts, ["http://a"], lease=8, seed=0)
    assert prompts.__getitem__.call_count == 512
