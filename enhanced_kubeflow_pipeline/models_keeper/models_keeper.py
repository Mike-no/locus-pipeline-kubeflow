import argparse
import sys
import shutil
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Output', type = str, help = "Path where openpflow trajectories file should be written.")
    parser.add_argument('--Output2', type = str, help = "Path where nodes.npz file should be written.")
    parser.add_argument('--Output3', type = str, help = "Path where tokyo_osmpbf.ft file should be written.")
    parser.add_argument('--Output4', type = str, help = "Path where tree.pkl file should be written.")
    args = parser.parse_args()

    if len(sys.argv) != 9:
        parser.print_help(sys.stderr)
        sys.exit(1)

    Path(args.Output).parent.mkdir(parents = True, exist_ok = True)
    Path(args.Output2).parent.mkdir(parents = True, exist_ok = True)
    Path(args.Output3).parent.mkdir(parents = True, exist_ok = True)
    Path(args.Output4).parent.mkdir(parents = True, exist_ok = True)

    shutil.copyfile('/models/e1.csv', args.Output)
    shutil.copyfile('/models/nodes.npz', args.Output2)
    shutil.copyfile('/models/tokyo_osmpbf.ft', args.Output3)
    shutil.copyfile('/models/tree.pkl', args.Output4)
