"""Export data_100 RGB encoding and unchanged gray16 depth; retain lossless source."""
import argparse
from beavr.lerobot.common.datasets.data100_export import export_data100

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(export_data100(args.source,args.output))

if __name__=='__main__': main()
