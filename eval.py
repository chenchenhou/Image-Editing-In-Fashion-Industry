from cleanfid import fid
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--real_folder", type=str, required=True, help="Path to the real image folder"
    )
    parser.add_argument(
        "--fake_folder", type=str, required=True, help="Path to the fake image folder"
    )
    return parser

parser = get_parser()
args = parser.parse_args()

real_folder = args.real_folder
fake_folder = args.fake_folder

score = fid.compute_fid(real_folder, fake_folder)
print(f"FID score: {score:.5f}")
