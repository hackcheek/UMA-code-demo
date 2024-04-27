import time
import docker
import re

IMAGE_NAME = 'py_executor'
anvil_endpoint = 'http://host.docker.internal:8545'
ipfs_host = 'host.docker.internal'

def logger(*args, **kwargs):
    print(*args, **kwargs)
    if 'ERROR' in str(args) or 'error:' in str(args).lower():
        raise RuntimeError('error')

def exec_code(code):
    filename = '/tmp/code.py'
    dest = "saracatunga"

    code = (
        f'import freewillai\n'
        f'freewillai.connect("devnet/anvil", "{anvil_endpoint}", ipfs_host="{ipfs_host}")\n'
        + code
    )

    with open(filename, 'w') as file:
        file.write(code)

    client = docker.from_env()
    container = client.containers.run(
        IMAGE_NAME, dest, detach=True, volumes=[f'{filename}:/repl/{dest}']
    )
    container_name = container.name
    last_logs = ''
    while True:
        container = client.containers.get(container_name)
        time.sleep(1)
        logs = "\n" + container.logs().decode('utf-8').lstrip(last_logs)
        logger(logs) if logs.strip() != '' else ...
        if container.status == 'exited':
            break
        last_logs = logs
    container.remove(force=True)


class TestREPL:
    def test_imports(self):
        with open('tests/imports.txt', 'r') as code:
            exec_code(code.read())

    def test_sklearn_demo(self):
        with open('demos/python/sklearn/sklearn_house_prediction_test.py', 'r') as code:
            exec_code(code.read())

    def test_pytorch_demo(self):
        with open('demos/python/pytorch/pytorch_titanic_predictor.py', 'r') as code:
            exec_code(code.read())

    def test_keras_demo(self):
        with open('demos/python/tensorflow/tensorflow_fraud_detection.py', 'r') as code:
            exec_code(code.read())
