import base58
import subprocess
import logging
import binascii
import re
import os
import asyncio
import aioipfs
import asyncio
import aioipfs
import eth_account
import torch
import numpy as np
import random
import zipfile
from web3 import Web3, HTTPProvider, WebsocketProvider
from web3.types import TxParams, BlockIdentifier, RPCEndpoint, RPCResponse
from tqdm import tqdm
from typing import (
    Coroutine, Dict, Iterable, List, Optional, Tuple, Literal, Any, Callable
)
from freewillai.converter import ModelConverter
from freewillai.doctypes import Abi, AddedFile, Bytecode, Middleware
from freewillai.exceptions import NotSupportedError, UserRequirement
from freewillai.globals import Global
from dotenv import load_dotenv
from eth_account.signers.local import LocalAccount
from solcx import compile_standard
from torch.nn import Module as PytorchModule
from tensorflow.keras import Model as KerasModule
from tempfile import NamedTemporaryFile
from pathlib import Path
from transformers import PreTrainedTokenizerBase
from typing import cast
from toolz import assoc
from functools import partial


import time

async def add_files(files: Iterable[str]) -> List[AddedFile]:
    t1 = time.perf_counter()
    async with aioipfs.AsyncIPFS(host=Global.ipfs_host, port=int(Global.ipfs_port)) as client:
        t2 = time.perf_counter()
        print("add_files HERE 1 took", t2-t1)
        lst = []
        async for added_file in client.add(*files, recursive=True):
            logging.debug('Imported file {0}, URL: {1}'.format(
                added_file['Name'], get_url(added_file['Hash'])))
            lst.append(added_file)
        t3 = time.perf_counter()
        print("add_files HERE 2 took", t3-t2)
        return lst            


async def get_file(cid: str, out_dir: Optional[str] = None) -> str:
    t1 = time.perf_counter()
    async with aioipfs.AsyncIPFS(host=Global.ipfs_host, port=int(Global.ipfs_port)) as client:
        out_dir = out_dir or Global.working_directory
        t2 = time.perf_counter()
        print("add_files HERE 1 took", t2-t1)
        await client.get(cid, out_dir)
        t3 = time.perf_counter()
        print("add_files HERE 2 took", t3-t2)
        return os.path.join(out_dir, cid)


def save_file(inp, filename, mode="wb"):
    if not os.path.exists(Global.working_directory):
        os.mkdir(Global.working_directory)
    out_path = os.path.join(Global.working_directory, filename)
    print(f"[*] Downloading on {out_path}")
    with open(out_path, mode) as file:
        file.write(inp)


def in_cache(cid_or_url: str) -> bool:
    if not os.path.exists(Global.working_directory):
        try:
            os.mkdir(Global.working_directory)
        except FileExistsError:
            ...
    if 'ipfs' in cid_or_url:
        cid_or_url = get_hash_from_url(cid_or_url)
    return cid_or_url in os.listdir(Global.working_directory)


def is_ipfs_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith('https://ipfs.io/ipfs/')


def get_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    return loop
    

def async_runner(*tasks: Coroutine) -> List:
    async def _main():
        global values
        values = await asyncio.gather(*tasks) 

    loop = get_loop()
    loop.run_until_complete(_main())
    return values


def get_hash_from_url(url: str) -> str:
    return re.findall(r'ipfs\/(.*)', url)[0]


def get_url(hsh: str) -> str:
    return f'https://ipfs.io/ipfs/{hsh}'


def get_path(cid_or_url: str) -> str:
    if not os.path.exists(Global.working_directory):
        os.mkdir(Global.working_directory)
    if 'https://ipfs.io/ipfs/' in cid_or_url:
        cid_or_url = get_hash_from_url(cid_or_url)
    return Global.working_directory + cid_or_url


def get_convertion_func(model):
    if isinstance(model, PytorchModule):
        return ModelConverter.pytorch_to_onnx

    elif isinstance(model, KerasModule):
        return ModelConverter.keras_to_onnx

    elif "<class 'sklearn" in repr(type(model)):
        return ModelConverter.sklearn_to_onnx

    elif 'onnx' in repr(type(model)):
        if 'InferenceSession' in repr(type(model)):
            raise NotSupportedError('Onnx model can\'t be an InferenceSession')
        return ModelConverter.dump_onnx

    else:
        raise NotSupportedError('Model library is not supported yet')


def get_modellib(model):
    from transformers import PreTrainedModel
    from sklearn.base import BaseEstimator
    from onnx import ModelProto
    lib_match = {
        isinstance(model, PytorchModule): 'torch',
        isinstance(model, KerasModule): 'keras',
        isinstance(model, PreTrainedTokenizerBase): 'tokenizer',
        isinstance(model, PreTrainedModel): 'huggingface',
        isinstance(model, BaseEstimator): 'sklearn',
        isinstance(model, ModelProto): 'onnx',
    }
    return lib_match.get(True) or None


def get_bytes32_from_hash(hsh: str):
    bytes_array = base58.b58decode(hsh)
    b = bytes_array[2:]
    hex = binascii.hexlify(b).decode('utf-8')
    return Web3.to_bytes(hexstr=hex)


def get_hash_from_bytes32(bytes_array) -> str:
    merged = 'Qm' + bytes_array
    return base58.b58encode(merged).decode('utf-8')


def get_account(private_key=None) -> LocalAccount:
    load_dotenv()
    if private_key is None:
        private_key = os.environ.get('PRIVATE_KEY')

    try:
        return eth_account.Account.from_key(private_key)

    except ValueError as err:
        raise ValueError(err)

    except:
        raise UserRequirement(
            "PRIVATE_KEY is required as environment variable.\n"
            "Please set your the private key by following one of these ways:\n"
            "  > executing a bash command: export PRIVATE_KEY='paste-your-private-key-here'\n"
            "  > Write into the .env file: PRIVATE_KEY='paste-your-private-key-here'"
        )


def compile_test_contract(
    contract_path: str, contract_name: str, settings: Optional[Dict] = None
) -> Tuple[Abi, Bytecode]:

    with open(contract_path, 'r') as contract_file:
        contract_code = contract_file.read()

    if settings is None:
        settings = {
            "language": "Solidity",
            "sources": {
                contract_path: {
                    "content": contract_code
                }
            },
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode"]
                    }
                }
            }
        }

    os.chdir('smart_contract/')
    compiled_contract = compile_standard(settings) 
    os.chdir('..')

    abi = compiled_contract['contracts'][contract_path][contract_name]['abi']
    bytecode = compiled_contract['contracts'][contract_path][contract_name]['evm']['bytecode']['object']
        
    return abi, bytecode


def get_w3(
    rpc_url: str,
    middlewares: List[Middleware] = [],
    modules: Optional[Any] = None,
):
    web3_provider_class = HTTPProvider
    kwargs = dict(request_kwargs={'timeout': 600})
    if rpc_url.startswith("ws"):
        web3_provider_class = WebsocketProvider
        kwargs = {}
    return Web3(
        web3_provider_class(rpc_url, **kwargs),
        middlewares,
        modules
    )


def load_global_env(file_path: str, override: bool = True) -> bool:
    loaded = load_dotenv(file_path, override=override)
    if not loaded:
        return loaded
    
    Global.update()

    return loaded


# zipping a directory, with a progress bar:
def zip_directory(directory_to_zip, zip_file_path):
    directory_to_zip = Path(directory_to_zip)
    file_list = list(directory_to_zip.rglob('*'))

    # Get the total size of the files to be zipped
    total_size = sum(file.stat().st_size for file in file_list)

    # Initialize the progress bar
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, mininterval=0.1, desc='Zipping')

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for item in file_list:
            if item.is_file():
                # Calculate the relative path for the file
                relative_path = item.relative_to(directory_to_zip)

                # Create a ZipInfo object with the appropriate metadata
                zip_info = zipfile.ZipInfo(str(relative_path))
                zip_info.compress_type = zipfile.ZIP_DEFLATED

                # Open the source file and a new entry in the zip file
                with open(item, 'rb') as src_file, zip_file.open(zip_info, 'w') as dest_file:
                    # Copy data from the source file to the zip file in chunks,
                    # updating the progress bar as we go
                    while True:
                        buf = src_file.read(1024 * 1024 * 1)  # Read 1MB chunks
                        if not buf:
                            break
                        dest_file.write(buf)
                        progress_bar.update(len(buf))

    # Close the progress bar
    progress_bar.close()



def zip_directory_old(directory_to_zip, zip_file_path):
    directory_to_zip = Path(directory_to_zip)
    file_list = list(directory_to_zip.rglob('*'))

    # Get the total size of the files to be zipped
    total_size = sum(file.stat().st_size for file in file_list)

    # Initialize the progress bar
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Zipping')

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file in file_list:
            zip_file.write(file, file.relative_to(directory_to_zip))

            # Update the progress bar based on the file size
            progress_bar.update(file.stat().st_size)
    progress_bar.close()


def generate_text(model, tokenizer, sequence, max_length):
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    set_seed(42)
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=1,
        top_p=1,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def deploy_contracts(out_env:str="anvil.contracts.env", private_key=None, rpc_url:str="http://127.0.0.1:8545"):
    arg = private_key or 'env'
    script_path = '/'.join(__file__.split("/")[:-2]) + "/scripts/deploy_contracts.sh"
    subprocess.check_call(f"{script_path} {arg} {rpc_url} {out_env}", shell=True)


def setup_anvil():
    subprocess.check_call(f"./scripts/setup_anvil.sh", shell=True)


def contract_exists(address: str, w3: Web3):
    return w3.eth.get_code(address).hex() != '0x'


def add_doc(docstring: str):
    """Add more documentation to existing"""
    def wrapper(func: Callable):
        func.__doc__ = str(func.__doc__) + '\n' + docstring
        return func
    return wrapper


def remove_path(path) -> None:
    if os.path.exists(path) and os.path.isdir(path):
        os.rmdir(path)
    elif os.path.exists(path):
        os.remove(path)


def run_tasks(*tasks:Coroutine):
    async def _gather():
        return await asyncio.gather(*tasks)
    asyncio.run(_gather())


def get_free_device():
    proc = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader'],
        stdout=subprocess.PIPE
    )
    stdout, _ = proc.communicate()

    free_mem = []
    for line in stdout.decode('utf-8').strip().split('\n'):
        device_id, mem = line.split(',')
        mem = mem.rstrip('MiB').strip()
        free_mem.append((int(device_id.strip()), int(mem) * 1024 * 1024))
    
    return max(free_mem, key=lambda x: x[1])


def get_free_mem_of_gpu(device:str):
    device_id = device.replace('cuda:', '').strip()
    proc = subprocess.Popen(
        ['nvidia-smi', '-i', device_id, '--query-gpu=memory.free', '--format=csv,noheader'],
        stdout=subprocess.PIPE
    )
    stdout, _ = proc.communicate()
    return stdout.decode('utf-8').strip().rstrip('MiB').strip()


def get_block_gas_limit(
    w3: "Web3", block_identifier: Optional[BlockIdentifier] = None
) -> int:
    if block_identifier is None:
        block_identifier = w3.eth.block_number
    block = w3.eth.get_block(block_identifier)
    return block["gasLimit"]


def get_buffered_gas_estimate(
    w3: "Web3", transaction: TxParams, gas_buffer: int = 100000
) -> int:
    gas_estimate_transaction = cast(TxParams, dict(**transaction))
    gas_estimate = w3.eth.estimate_gas(gas_estimate_transaction)
    gas_limit = get_block_gas_limit(w3)

    if gas_estimate > gas_limit:
        raise ValueError(
            "Contract does not appear to be deployable within the "
            f"current network gas limits.  Estimated: {gas_estimate}. "
            f"Current gas limit: {gas_limit}"
        )

    return min(gas_limit, gas_estimate + gas_buffer)


def pickleable_middleware(method: RPCEndpoint, params: Any, make_request, w3) -> RPCResponse:
    if method == "eth_sendTransaction":
        transaction = params[0]
        if "gas" not in transaction:
            transaction = assoc(
                transaction,
                "gas",
                hex(get_buffered_gas_estimate(w3, transaction)),
            )
            return make_request(method, [transaction])
    return make_request(method, params)


def buffered_gas_estimate_middleware(
    make_request: Callable[[RPCEndpoint, Any], Any], w3: "Web3"
) -> Callable[[RPCEndpoint, Any], RPCResponse]:
    """
    Pickleable gas estimate middleware
    """
    return partial(pickleable_middleware, make_request=make_request, w3=w3)
