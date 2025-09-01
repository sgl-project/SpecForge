import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.output_path, "w") as out_f:
        with open(args.input_path, "r") as in_f:
            for line in in_f:
                data = json.loads(line)
                new_data = dict(conversations=[])

                for message in data["conversations"]:
                    new_data["conversations"].append(
                        dict(role=message["from"], content=message["value"])
                    )
                out_f.write(json.dumps(new_data) + "\n")


if __name__ == "__main__":
    main()
