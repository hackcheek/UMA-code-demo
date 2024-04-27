from __future__ import annotations

import os
import time
import asyncio
import tensorflow as tf
import torch
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple, Dict, List, Type, Union, Literal, Any
from web3.types import Middleware as Web3Middleware
from eth_account.signers.local import LocalAccount
from freewillai.common import (
    ContractNodeResult, IPFSBucket, FWAIModel, IPFSFile,
    FWAIDataset, Task, FWAIResult, Network, Middleware, FWAIModel
)
from freewillai.doctypes import BaseTokenizer
from freewillai.globals import Global
from freewillai.exceptions import FreeWillAIException
from freewillai.utils import (
    get_account, get_hash_from_url, get_url,
    is_ipfs_url, load_global_env, remove_path
)
from freewillai.contract import TaskRunnerContract, TokenContract


class TaskRunner:
    def __init__(
        self, 
        model, 
        dataset,
        min_time: int = 1,
        max_time: int = 10, #200,
        min_results: int = 2,
        tokenizer = None,
        preprocess: Optional[Dict] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        private_key_or_account: Optional[Union[str, LocalAccount]] = None,
        poll_interval: int = 1,
        network: Union[Network, str, None] = None,
        force_validation: bool = False,
        is_generative: bool = False,
        model_kwargs: Dict = {},
        tokenizer_kwargs: Dict = {},
    ):
        self.min_results = 3
        print(
            "[!] We recommend testing this task locally "
            "before running it on the blockchain to avoid any issues\n"
        )

        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.min_time = min_time
        self.max_time = max_time
        self.min_results = min_results
        self.preprocess = preprocess
        self.input_size = input_size
        self.poll_interval = poll_interval
        self.force_validation = force_validation
        self.is_generative = is_generative
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        self.model_path = NamedTemporaryFile().name
        self.dataset_path = NamedTemporaryFile().name
        
        self.fwai_dataset = FWAIDataset(dataset) 
        self.fwai_model = FWAIModel(
            model=model,
            input_format=self.fwai_dataset.format(),
            tokenizer=tokenizer,
            input_size=input_size,
            preprocess=preprocess,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=model_kwargs,
        )

        if isinstance(private_key_or_account, LocalAccount):
            self.account = private_key_or_account
        else:
            self.account = get_account(private_key_or_account)
        
        if isinstance(network, str):
            network = Network(network)
        self.network = network or Global.network or Network.build()

        self.network.allow_sign_and_send(self.account)
            
        self.token = TokenContract(
            self.account, 
            address=self.network.token_address,
            network=self.network
        )
        self.task_runner = TaskRunnerContract(
            self.account, 
            address=self.network.task_runner_address,
            network=self.network,
            token_contract=self.token
        )

        self.model_cid = None
        self.dataset_cid = None
        self.result = None
        self.start_time = 0


    async def dispatch(self) -> None:
        if not is_ipfs_url(self.dataset):
            dataset_path = self.fwai_dataset.save()
            if not os.path.exists(self.dataset_path):
                RuntimeError(
                    'Unexpected error in the existence of the dataset path. '
                    f'dataset_path={self.dataset_path}'
                )
            ipfs_dataset = await IPFSBucket.upload(dataset_path, 'dataset') 
            self.dataset_cid = ipfs_dataset.cid

        else:
            self.dataset_cid = get_hash_from_url(self.model)

        if not is_ipfs_url(self.dataset):
            model_path = self.fwai_model.save()

            if not os.path.exists(self.model_path):
                RuntimeError(
                    'Unexpected error in the existence of the model path. '
                    f'model_path={self.model_path}'
                )
            ipfs_model = await IPFSBucket.upload(model_path, 'model') 
            self.model_cid = ipfs_model.cid

        else:
            self.model_cid = get_hash_from_url(self.model)

        # Remove files
        remove_path(self.model_path)
        remove_path(self.dataset_path)


    async def listen_and_validate_response(self, task) -> FWAIResult:
        if Global.verbose:
            print(f"\n"
                f"[*] Waiting for result. "
                f"minTime={self.min_time}s "
                f"maxTime={self.max_time}s "
                f"minResults={self.min_results}"
            )

        print('\n')
        bar = tqdm(
            # total=self.max_time, 
            total=1, 
            bar_format='{percentage:3.0f}%|{bar:20}| {remaining} left{desc}'
        )

        sleep_seconds = 0.1 if self.network.id == 'devnet/anvil' else .01
        base_time = time.perf_counter()
        final_progress = 0
        last_timestamp = base_time
        results = []
        velocity = 0
        factor = 1 if self.network.id == 'devnet/anvil' else .1
        nodes_str = ''
        n_results = 0

        def _linear_interpolate(start, end, point):
            return start + (end - start) * point

        while time.perf_counter() - base_time < self.max_time:
            timestamp = time.perf_counter()
            delta_time = timestamp - last_timestamp

            await asyncio.sleep(sleep_seconds)

            results: List[ContractNodeResult] = (
                self.task_runner.get_available_task_results(task.id, log=False)
            ) if len(results) < self.min_results else results

            node_progress = len(results) / self.min_results
            time_progress = (time.perf_counter() - base_time) / self.max_time
            last_final_progress = final_progress
            combined_progress = _linear_interpolate(node_progress, 1, time_progress)
            velocity += (combined_progress - last_final_progress) * factor * delta_time 
            final_progress = abs(last_final_progress + velocity * delta_time)

            len_results = len(results)
            if len_results > n_results:
                for result in results[n_results:]:
                    bar.write(
                        f"[*] Received validation from a node:\n"
                        f"  > result: {result.url}\n"
                        f"  > sender: {result.sender}"
                    )
                    n_results = len_results

            bar.n = min(final_progress, 1)
            bar.refresh()

	        ### TODO: remove it [testing function]
            if self.network.id == 'devnet/anvil':
                self.task_runner.generate_block()

            if final_progress >= 1 and self.task_runner.check_if_ready_to_validate(task.id, log=False):
                break        

            last_timestamp = timestamp
            
        bar.close()
        print('\n')
        
        return await self.validate(task)


    async def validate(self, task) -> FWAIResult:
        print('  > Validating results...')

        t1 = time.perf_counter()
        tx_hash = self.task_runner.validate_task_if_ready(task.id)
        self.task_runner.wait_for_transaction(tx_hash)
        t2 = time.perf_counter()

        # TODO: develop a timer handler as an option to select
        # print(f"[TIMER] validate_task_if_ready took: {t2-t1:.3f}")

        result_url = self.task_runner.get_task_result(task.model_url, task.dataset_url)
        
        if result_url == "":
            import sys
            sys.tracebacklimit = 0
            raise FreeWillAIException(
                    "No consensus reached: not enough nodes agreed on a correct AI result. Please try again with a higher maxTime or lower minResults to allow the network to reach consensus. "
                "You can also check with Free Will AI support about the network status"
            ) from None
        return await self.get_result(result_url)


    async def get_result(self, result_url):
        await IPFSBucket.download(result_url, file_type='result')
        self.result = FWAIResult.load(result_url)
        return self.result


    async def run_and_get_result(self) -> FWAIResult:
        assert self.model_cid and self.dataset_cid
        model_url = get_url(self.model_cid)
        dataset_url = get_url(self.dataset_cid)

        if not self.force_validation:
            # Get result if it's already validated
            maybe_result_url = self.task_runner.get_task_result(model_url, dataset_url)
            if maybe_result_url:
                return await self.get_result(maybe_result_url)

        t1 = time.perf_counter()
        _, event_data = self.task_runner.add_task(
            model_url=model_url,
            dataset_url=dataset_url,
            min_time=self.min_time,
            max_time=self.max_time,
            min_results=self.min_results
        )
        t2 = time.perf_counter()

        task = Task.load_from_event(event_data, self.network)
        t3 = time.perf_counter()
        result = await self.listen_and_validate_response(task)
        t4 = time.perf_counter()
        # TODO: develop a timer handler as an option to select
        # print(f"[TIMER] self.task_runner.add_task took: {t2-t1:.3f}")
        # print(f"[TIMER] Task.load_from_event took: {t3-t2:.3f}")
        # print(f"[TIMER] self.listen_and_validate_response took: {t4-t3:.3f}")
        return result


async def run_task(
    model,
    dataset, 
    min_time: int = 1,
    max_time: int = 200,
    min_results: int = 2,
    tokenizer = None,
    preprocess: Optional[Dict] = None, 
    input_size: Optional[Tuple[int, ...]] = None,
    verbose: bool = False,
    private_key_or_account: Optional[Union[str, LocalAccount]] = None,
    network: Optional[Network] = None,
    force_validation: bool = False,
    return_tensor: Optional[Literal['np', 'pt', 'tf']] = None,
    model_kwargs: Dict = {},
    tokenizer_kwargs: Dict = {},
) -> FWAIResult:
    assert return_tensor in ['np', 'pt', 'tf', None]
    if verbose:
        Global.verbose = True

    if os.environ.get('DEMO'):
        force_validation = True

    t0 = time.perf_counter()
    tr = TaskRunner(
        model, 
        dataset, 
        min_time,
        max_time,
        min_results,
        tokenizer,
        preprocess, 
        input_size, 
        network=network,
        force_validation=force_validation,
        private_key_or_account=private_key_or_account,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    t1 = time.perf_counter()
    await tr.dispatch()
    t2 = time.perf_counter()

    # Timers
    # print(f"[DEBUG][core.py] TaskRunner.__init__: {t1-t2:.3f}")
    # print(f"[DEBUG][core.py] dispatch took: {t2-t1:.3f}")
    result = await tr.run_and_get_result()

    if isinstance(result, str):
        if return_tensor:
            print(f"[!] Returning string instead of {return_tensor=} due to nlp model")
        return result

    return_tensor = return_tensor or 'np'
    to_tensor = {
        'np': lambda _: _,
        'pt': torch.from_numpy,
        'tf': tf.convert_to_tensor,
    }
    return to_tensor[return_tensor](result)


async def upload_dataset(dataset: Any) -> IPFSFile:
    """
    Function to upload a dataset to ipfs checked for freewillai

    Arguments
    ---------
    dataset: Any dataset
    """
    dest_path = FWAIDataset(dataset).save()
    return await IPFSBucket.upload(dest_path, 'dataset')


async def upload_model(
    model: Any,
    input_format: str,
    tokenizer: Optional[Union[BaseTokenizer, str]] = None,
    input_size: Optional[Tuple[int, ...]] = None,
    preprocess: Optional[Dict] = None,
    model_kwargs: Dict = {},
    tokenizer_kwargs: Dict = {},
) -> IPFSFile:
    """
    Convert model to non-turing and upload it to ipfs
    """
    dest_path = FWAIModel(
        model,
        input_format,
        tokenizer,
        input_size,
        preprocess,
        model_kwargs,
        tokenizer_kwargs,
    ).save()
    return await IPFSBucket.upload(dest_path, 'model')


def connect(
    network_id_or_rpc_url:Union[str, Network],
    custom_rpc:Optional[str]=None,
    middlewares: List[Union[Web3Middleware, Middleware]] = [],
    token_address: Optional[str] = None,
    task_runner_address: Optional[str] = None,
    env_file: Optional[str] = None,
    ipfs_host: Optional[str] = None,
    ipfs_port: Optional[int] = None,
) -> Union[Network, Type[Network]]:
    env_file = env_file or ".env"
    load_global_env(env_file)

    if isinstance(network_id_or_rpc_url, Network):
        network = network_id_or_rpc_url
    else:
        network = Network(
            network_id_or_rpc_url,
            middlewares,
            token_address,
            task_runner_address
        )

    if not custom_rpc is None:
        network = network.with_custom_rpc(custom_rpc)

    network.connect()

    Global.network = network

    if ipfs_host:
        os.environ['IPFS_HOST'] = ipfs_host
        Global.ipfs_host = ipfs_host
    if ipfs_port:
        os.environ['IPFS_PORT'] = str(ipfs_port)
        Global.ipfs_port = ipfs_port
    return network
