import sys
from freewillai.common import Anvil


if __name__ == "__main__":
    assert len(sys.argv) == 3, "\nUsage: python get_anvil_envfile.py config.json outfile.env"

    file_path = sys.argv[1]
    out_path = sys.argv[2]

    assert file_path.endswith('.json'), "The argument must be an json file and ends with .json"

    Anvil(file_path).build_envfile(out_path)
