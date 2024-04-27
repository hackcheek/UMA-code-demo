import sys
import json


def cp_abi(in_path, out_path):
    # copy just abi key
    with open(in_path, 'r') as in_file:
        data = json.loads(in_file.read())

    print(data['abi'])

    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(data['abi'], indent=4))


if __name__ == "__main__":
    contract = sys.argv[1]
    paths = {
        "taskrunner": ["./out/TaskRunner.sol/TaskRunner.json", "./contracts/TaskRunnerABI.json"],
        "token": ["./out/FreeWIllAIToken.sol/FreeWillAI.json", "./contracts/FreeWillAITokenABI.json"]
    }
    
    in_path, out_path = paths[contract.lower()]
    cp_abi(in_path, out_path)
