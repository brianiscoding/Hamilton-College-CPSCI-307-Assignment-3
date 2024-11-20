import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()

test_data_path = args.input
