from argparse import ArgumentParser
from pathlib import Path

TRAIN_COUNT = 340_000
TRAIN_CHUNK_SIZE = 170_000
TEST_COUNT = 10_000


def split_jsonl(src_path: Path, train_path: Path, test_path: Path) -> None:
    train_dir = train_path.parent
    if train_dir:
        train_dir.mkdir(parents=True, exist_ok=True)

    train_suffix = train_path.suffix or ".jsonl"
    train_stem = train_path.stem if train_path.suffix else train_path.name

    def next_train_chunk_path(index: int) -> Path:
        return train_dir / f"{train_stem}{index}{train_suffix}" if train_dir else Path(
            f"{train_stem}{index}{train_suffix}"
        )

    train_written = 0
    test_written = 0
    chunk_written = 0
    chunk_index = 1
    train_out = None

    try:
        with src_path.open("r", encoding="utf-8") as src, test_path.open(
            "w", encoding="utf-8"
        ) as test_out:
            for line in src:
                if train_written < TRAIN_COUNT:
                    if train_out is None:
                        train_out = next_train_chunk_path(chunk_index).open(
                            "w", encoding="utf-8"
                        )
                    train_out.write(line)
                    train_written += 1
                    chunk_written += 1

                    if chunk_written == TRAIN_CHUNK_SIZE and train_written < TRAIN_COUNT:
                        train_out.close()
                        train_out = None
                        chunk_written = 0
                        chunk_index += 1
                    continue

                if test_written < TEST_COUNT:
                    test_out.write(line)
                    test_written += 1

                    if test_written == TEST_COUNT:
                        break

    finally:
        if train_out is not None:
            train_out.close()

    if train_written < TRAIN_COUNT or test_written < TEST_COUNT:
        raise ValueError(
            f"{src_path} has only {train_written + test_written} usable lines;"
            f" need {TRAIN_COUNT} for train and {TEST_COUNT} for test."
        )


def main() -> None:
    parser = ArgumentParser(description="Split a JSONL file into train/test segments.")
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Source JSONL file",
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        required=True,
        help="Output path for training subset",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        required=True,
        help="Output path for testing subset",
    )
    args = parser.parse_args()

    split_jsonl(args.input_path, args.train_path, args.test_path)


if __name__ == "__main__":
    main()
