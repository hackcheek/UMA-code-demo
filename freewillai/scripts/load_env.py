#! /usr/bin/env python3

import argparse
import os
from freewillai.constants import FWAI_DIRECTORY
from freewillai.utils import save_file


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--env-file", type=str)
    parser.add_argument("-e", "--export", action='append', type=str)
    parser.add_argument("-v", "--verbose", action=argparse.BooleanOptionalAction, type=bool)

    args = parser.parse_args()

    usage_msg = "Usage: load_env.py -f path/to/env_file -e var1=value1 -e var2=value2"
    out_env_string = ""
    
    if not args.env_file and not args.export:
        print("[?] Nothing to do")
        print("  >", usage_msg)
        return
    
    if args.env_file:
        with open(args.env_file, "r") as env_file:
            out_env_string += env_file.read()
        print(f"\n[*] loaded {args.env_file}")

    if args.export:
        print("\n[*] Exporting variables to env")
        out_env_string += "\n".join(args.export)

    if args.verbose:
        print(f"\nFile:\n{out_env_string}")

    save_file(out_env_string, "global.env", mode="w")


if __name__ == "__main__":
    cli()
