import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Input', type = str, help = "URL of the file to be downloaded.")
    parser.add_argument('--Output', type = str, help = "Path of the local file where the file should be written.")
    args = parser.parse_args()

    if len(sys.argv) != 5:
        parser.print_help(sys.stderr)
        sys.exit(1)

    Path(args.Output).parent.mkdir(parents = True, exist_ok = True)

    file_name = args.Input.split('/')[-1]
    print("Downloading: %s." % (file_name))

    u = urlopen(args.Input)
    f = open(args.Output, 'wb')
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break;

        f.write(buffer)

    f.close()
    print("Done.")