from __future__ import annotations
from asyncio.futures import Future
import abc
import signal
import os
import json
import multiprocess
import yaml
import asyncio
import random
import numpy as np
import time
import argparse
import traceback
from requests.exceptions import HTTPError
from typing import Iterable, List, Tuple, Union, Dict, cast, Iterable
from tempfile import NamedTemporaryFile
from dotenv import dotenv_values
from concurrent.futures import ProcessPoolExecutor
from freewillai.common import (
    Anvil, FWAIDataset, FileDetector, IPFSBucket, Network, FWAIModel, Task, NonTuringModel
)
from freewillai.globals import Global
from freewillai.utils import get_account, get_url, in_cache, load_global_env, run_tasks
from freewillai.contract import TaskRunnerContract, TokenContract
from freewillai.exceptions import Fatal, NotSupportedError
from dataclasses import dataclass
from numpy import ndarray
from typing import Optional
from web3.exceptions import MethodUnavailable
from web3 import AsyncWeb3
from web3.providers import WebsocketProviderV2


@dataclass
class NodeConfig:
    private_key: str
    networks: List[Union[Network, str]]
    stake: int
    cooling_time: float = 0
    log_path: Optional[str] = None
    gpu: Optional[str] = None
    name: Optional[str] = None

    @staticmethod
    def load_pk_from_env(env_path:str) -> str:
        dot = dotenv_values(env_path)
        return dot.get("PRIVATE_KEY")

    @classmethod
    def from_dict(cls, conf:Dict):
        assert (
            (conf.get('private_key') or conf.get('env_file'))
            and conf.get('networks') and conf.get('stake')
        )
        private_key = conf.get('private_key')
        if private_key is None:
            private_key = cls.load_pk_from_env(conf.get('env_file'))

        nets = cast(List[Dict[str, Union[Dict, str]]], conf.get('networks'))
        networks = []
        for net in nets:
            if isinstance(net, dict):
                id, net_conf = net.copy().popitem()
                network = Network(id)
                if net_conf.get('rpc_url'):
                    network = network.with_custom_rpc(net_conf.get('rpc_url'))
                networks.append(network)
                continue

            networks.append(Network(net))

        log_path = conf.get('log_path')
        gpu = conf.get('gpu')
        name = conf.get('name')
        stake = int(conf.get('stake'))
        cooling_time = float(conf.get('cooling_time')) or 0
        return cls(
            private_key=private_key,
            networks=networks,
            stake=stake,
            cooling_time=cooling_time,
            log_path=log_path,
            gpu=gpu,
            name=name
        )
    
    def __repr__(self):
        return (
            str(type(self))[:-1] + "\n"
                f"    networks={self.networks},\n"
                f"    stake={self.stake},\n"
                f"    cooling_time={self.cooling_time},\n"
                f"    log_path={self.log_path}\n"
            ">"
        )


### TODO: Move to a new file and refactor
class EventListener:
    def __init__(self, node: Node, polling_time: float = 0.1): ...

    def __aiter__(self) -> EventListener:
        return self
    
    async def __anext__(self) -> Iterable[Dict]:
        raise NotImplementedError

    async def post_process(self, events: Iterable[Dict]) -> Iterable[Dict]:
        return events


class WebSocketListener(EventListener):
    def __init__(self, node: Node, polling_time: float = 0.1):
        self.node = node
        self.polling_time = polling_time
        self.w3 = AsyncWeb3.persistent_websocket(WebsocketProviderV2(node.network.rpc_urls[0]))
        loop = asyncio.get_running_loop()
        self.subscription_id = None
        self._setup = loop.create_task(self.setup())
        self.events = []
        
    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.w3.ws.listen_to_websocket().__anext__()
        return [cast(Dict, event)]

    async def setup(self):
        if self._setup and self._setup.done():
            return self
        if not await self.w3.is_connected():
            await self.w3.provider.connect()
        self.subscription_id = await self.w3.eth.subscribe(
            "logs", {
                "address": self.node.task_runner.address,
                "topics": [self.w3.keccak(text="TaskAdded(uint256,string,string)")]
            }
        )
        return self

    def _parse_receipt(self, receipt) -> dict:
        event_data = self.node.task_runner.contract.events.TaskAdded().process_receipt(receipt)[0]
        return event_data

    async def post_process(self, events: Iterable[Dict]) -> Iterable[Dict]:
        tx_hashes = set(event["result"]["transactionHash"] for event in events)
        receipts = await asyncio.gather(*[
            self.w3.eth.get_transaction_receipt(tx_hash) for tx_hash in tx_hashes
        ])
        return [self._parse_receipt(receipt) for receipt in receipts if receipt]


class HTTPListener(EventListener):
    def __init__(self, node: Node, polling_time: float | None = None):
        self.task_runner = node.task_runner
        self.network = node.network
        self.node = node
        self.polling_time = polling_time or 0.1
        if self.network.id == "devnet/anvil":
            self.polling_time = max(self.polling_time, 0.5)
        self.event_filter = self.get_event_filter()
        self.last_block_scanned = self.latest_block()

    def __aiter__(self):
        return self

    async def __anext__(self):
        #self.node.log(f"Polling, pid: {os.getpid()}")
        await asyncio.sleep(self.polling_time)
        loop = asyncio.get_event_loop()
        # Recap block just in case
        self.last_block_scanned = self.latest_block()
        return await loop.run_in_executor(None, self.event_filter)

    def latest_block(self, subtract: int = 20):
        return max(self.task_runner.w3.eth.block_number - subtract, 1)

    def _event_filter_by_get_logs(self, from_block=None):
        # If from_block=None scan the last 5 blocks
        from_block = from_block or self.latest_block()
        params = {"fromBlock": from_block, "toBlock": 'latest'}
        logs = self.task_runner.contract.events.TaskAdded.get_logs(**params)

        # Doble check to address some network issues that gets logs before 
        # than block number filtering
        for event in logs:
            if event["blockNumber"] > from_block:
                yield event

    def _event_filter_by_create_filter(self, from_block=None):
        # If from_block=None scan the last 5 blocks
        from_block = from_block or self.latest_block()
        return (
            self.task_runner.contract.events.TaskAdded
            .create_filter(fromBlock=from_block, toBlock='latest')
        ).get_new_entries()

    def get_event_filter(self):
        try:
            if self.network.id in ["testnet/goerli", "testnet/sepolia"]:
                raise NotSupportedError
            self._event_filter_by_create_filter()
            return self._event_filter_by_create_filter
        
        except (MethodUnavailable, ValueError, NotSupportedError, HTTPError) as err:
            # self.log(f"[DEBUG] Error in get_event_filter net={self.network.id}: {err}")
            # This is for nets like goerli that does not allow event filtering
            return self._event_filter_by_get_logs

        except Exception as err:
            raise Fatal(err)


class Node:
    def __init__(
        self,
        # TODO: Multiprocessing
        cores: int = 1,
        private_key: Optional[str] = None,
        test_bad_result: bool = False,
        test_bad_chance: Optional[float] = None,
        anvil: Optional[Anvil] = None,
        network: Union[Network, str, None] = None,
        cooling_time: float = 0,
        log_path: Optional[str] = None,
        gpu: Optional[str] = None,
        name: Optional[str] = None
    ):
        self.private_key = private_key
        self.test_bad_result = test_bad_result
        self.test_bad_chance = test_bad_chance

        self.log_path = log_path
        self.gpu = gpu
        self.name = name or 'worker'

        self.cached_models = {}
        
        if isinstance(network, str):
            network = Network(network)
        self.network = network or Network.build()

        self.account = get_account(self.private_key)
        self.network.allow_sign_and_send(self.account)

        self.address = self.account.address
        self.token = TokenContract(self.account, network=self.network)
        self.task_runner = TaskRunnerContract(
            self.account, 
            network=self.network, 
            token_contract=self.token
        )
        self.anvil = anvil

        # List of task running now
        self.running_tasks: Dict[int, Future] = {}

        # List of pending to validate tasks
        self.pending_tasks: List[int] = []

        # List of tasks that raise errors
        self.blacklist: List[int] = []

        self.cooling_time = cooling_time

        if not self.token.fwai_balance > 0:
            raise Fatal(
                ## TODO: Use self.task_runner.staking_minimum instead of harcoded 100 
                ## after deploy task runner contract on sepolia
                f"You have {self.token.fwai_balance} FWAI in your wallet. "
                f"Please buy FWAI enough to stake (100 FWAI at least)"
            )

        self.staking_amount: int = self.task_runner.get_staking_amount(self.address)
        self.log_time = time.perf_counter()

        self.log(f'[*] Account Info')
        self.log(f'  > PID: {os.getpid()}')
        self.log(f'  > Your public key is {self.address}')
        self.log(f'  > ETH in account: {self.token.eth_balance}')
        self.log(f'  > FWAI in account: {self.token.fwai_balance}')


    @classmethod
    def from_config(cls, conf: NodeConfig):
        cls(
            private_key=conf.private_key,
            network=conf.networks[0],
            cooling_time=conf.cooling_time,
            log_path=conf.log_path,
            gpu=conf.gpu,
            name=conf.name,
        ).stake(conf.stake)


    def log(self, *args, **kwargs):
        """TEMP Logger"""
        prefix = f"[{self.name} on {self.network.id} : {time.perf_counter() - self.log_time:.2f}s]"
        print(prefix, *args, **kwargs)
        if self.log_path is None:
            return
        log_dir = os.path.split(self.log_path)[0]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        kwargs.pop('file', None)
        with open(self.log_path, 'a+') as file:
            print(prefix, *args, file=file, **kwargs)

        self.log_time = time.perf_counter()
        

    def stake(self, amount):
        self.staking_amount = self.task_runner.get_staking_amount(self.address)
        if amount > self.token.fwai_balance:
            raise Fatal(
                f"Staking amount ({amount} FWAI) "
                f"exceeds your balance ({self.token.fwai_balance} FWAI). "
                f"Please buy FWAI to stake"
            )

        if self.staking_amount < amount:
            tx_hash = self.token.approve(self.task_runner.address, amount)
            self.token.wait_for_transaction(tx_hash)
            tx_hash = self.task_runner.stake(amount - self.staking_amount)
            # This is just for print
            self.token.wait_for_transaction(tx_hash)

        elif self.staking_amount > amount:
            tx_hash = self.task_runner.unstake(self.staking_amount - amount)
            # This is just for print
            self.task_runner.wait_for_transaction(tx_hash)
            
        self.staking_amount = self.task_runner.get_staking_amount(self.address)
        self.log(f'  > Staking amount: {self.staking_amount}')
        return self


    async def run_task(self, task):
        t1 = time.perf_counter()
        tasks = []
        if not in_cache(task.model_url):
            self.log(f"[*] Downloading model {task.model_url}")
            tasks.append(IPFSBucket.download(task.model_url, file_type='model'))
        if not in_cache(task.dataset_url):
            self.log(f"[*] Downloading dataset {task.dataset_url}")
            tasks.append(IPFSBucket.download(task.dataset_url, file_type='dataset'))

        # Download both on simultaneously
        await asyncio.gather(*tasks)

        self.log('[*] Running Inference')

        model: NonTuringModel = FWAIModel.load(task.model_url, self.gpu)
        dataset: ndarray = FWAIDataset.load(task.dataset_url)
        result: ndarray = model.inference(dataset)

        if ((not self.test_bad_chance is None 
            and random.random() < self.test_bad_chance)
            or self.test_bad_result
        ):
            result = np.array([1])

        self.log('  > Inference Done\n')

        t2 = time.perf_counter()
        self.log(f"[*] Task took {t2-t1:.3f} seconds")
        await self.submit_result(task, result)


    async def submit_result(self, task, result):
        self.log('[*] uploading result to ipfs, task id:', task.id)
        ## TODO: choice between csv or numpy to upload a result 
        temp_result_file = NamedTemporaryFile().name # + '.npy'
        # np.save(temp_result_file, result)
        if isinstance(result, str):
            with open(temp_result_file, "w") as resultfile:
                resultfile.write(result)
        else:
            np.savetxt(temp_result_file, result, delimiter=";")
        ipfs_result = await IPFSBucket.upload(temp_result_file, file_type='result')
        cid = ipfs_result.cid

        self.log('[*] submitting result:', task.id)
        self.log('  > cid:', cid)

        # Submit result just can be processed one by one 
        # between all nodes in NeonEVM
        # TODO: Contact to Neon EVM support
        if self.network.id == "mainnet/neonevm":
            # Try submit result recursively
            def try_submit_result():
                try:
                    task.submit_result(get_url(cid), self.task_runner)
                except ValueError:
                    time.sleep(1) 
                    try_submit_result()
        else:
            task.submit_result(get_url(cid), self.task_runner)

        self.pending_tasks.append(task.id)


    def latest_block(self):
        return self.task_runner.w3.eth.block_number

    
    def _event_filter_by_get_logs(self, from_block=None):
        # If from_block=None scan the last 5 blocks
        from_block = from_block or self.latest_block() - 5
        params = {"fromBlock": from_block, "toBlock": 'latest'}
        logs = self.task_runner.contract.events.TaskAdded.get_logs(**params)

        # Doble check to address some network issues that gets logs before 
        # than block number filtering
        for event in logs:
            if event["blockNumber"] > from_block:
                yield event
    

    def _event_filter_by_create_filter(self, from_block=None):
        # If from_block=None scan the last 5 blocks
        from_block = from_block or self.latest_block() - 5
        return (
            self.task_runner.contract.events.TaskAdded
            .create_filter(fromBlock=from_block, toBlock='latest')
        ).get_new_entries()


    def get_event_filter(self):
        try:
            if self.network.id in ["testnet/goerli", "testnet/sepolia"]:
                raise NotSupportedError
            self._event_filter_by_create_filter()
            return self._event_filter_by_create_filter
        
        except (MethodUnavailable, ValueError, NotSupportedError, HTTPError) as err:
            # self.log(f"[DEBUG] Error in get_event_filter net={self.network.id}: {err}")
            # This is for nets like goerli that does not allow event filtering
            return self._event_filter_by_get_logs

        except Exception as err:
            raise Fatal(err)

    def sync_run_task(self, task):
        return asyncio.run(self.run_task(task))

    async def listen_for_event(self):
        event_listener = (HTTPListener(self, self.cooling_time) 
            if not self.network.rpc_urls[0].startswith("ws")
            else await WebSocketListener(self, self.cooling_time).setup()
        )
        async for events in event_listener:
            events = await event_listener.post_process(events)
            for event in events:
                # self.log("[DEBUG] event:", event)
                # See task id to fast conditional handle
                task_id: int = event['args']['taskIndex']

                if task_id in self.blacklist or task_id in self.running_tasks.keys():
                    continue

                if (task_id in self.pending_tasks
                    or self.task_runner.is_validated(task_id, log=False)
                    or self.task_runner.is_in_timeout(task_id, log=False)
                ):
                    continue

                staking_enough = self.task_runner.get_staking_amount(
                    self.address, log=False
                ) >= self.task_runner.staking_minimum

                if not staking_enough:
                    self.log("[!] Insufficient staking amount to run found task")
                    await asyncio.sleep(1)
                    break                        

                task = Task.load_from_event(event, self.network)
                try:
                    self.log(f'[*] Task found: {task}')
                    # proc = multiprocess.Process(target=self.sync_run_task, args=(task,))
                    # proc.start()
                    await self.run_task(task)
                    # self.running_tasks.update({task.id: proc})
                except:
                    self.log(f"[!] Error has been detected running task id={task.id}\n")
                    self.log(traceback.format_exc(chain=False))
                    self.log("-"*50, "\n")
                    self.blacklist.append(task.id)
                    continue

        
        # Pending tasks handling
        for idx, task_id in enumerate(self.pending_tasks):
            if (self.task_runner.is_in_timeout(task_id, log=False)
                or self.task_runner.is_validated(task_id, log=False)
            ):
                self.log(f'[*] Task {task_id} done')
                self.pending_tasks.pop(idx)

            # Maybe validate if ready.
            # if self.task_runner.check_if_ready_to_validate(pending_task.id):
            #     self.task_runner.validate_task_if_ready(pending_task.id)
        
        _running_tasks = self.running_tasks.items()
        for task_id, proc in _running_tasks:
            if proc.is_alive():
                del self.running_tasks[task_id]


    async def spin_up(self) -> None:
        import traceback
        self.log(f'[*] Spining up the node')
        try:
            await self.listen_for_event()

        except KeyboardInterrupt:
            self.log(traceback.format_exc(chain=False))

        except Fatal as err:
            raise Fatal(err)

        except Exception as err:
            self.log(traceback.format_exc(chain=False))
            return await self.spin_up()


        finally:
            self.log('[!] Node Killed')


class Runner:
    def __init__(self, configs: List[NodeConfig]):
        self.configs = configs 
        self.process: List[multiprocess.Process] = []

    async def _try_spin_up(self, node, num_retry: int, max_retries:int):
        if max_retries < num_retry:
            return
        try:
            await node.spin_up()
        except Fatal as err:
            raise Fatal(err)

        except Exception:
            err_path = f"/tmp/freewillai-error-in-{num_retry}"
            with open(err_path, "w") as err_file:
                err_file.write(traceback.format_exc(chain=False))
            node.log(f"[!] Retrying due to an unexpected error in iteration {num_retry}/{max_retries}")
            node.log(f"  > To see the complete log of the error=`cat {err_path}`\n")
            with open(err_path, 'r') as f:
                node.log(f.read())
            return await self._try_spin_up(node, num_retry+1, max_retries)

    async def start_node(self, conf, network):
        max_retries = 5
        node = Node(
            private_key=conf.private_key,
            network=network,
            cooling_time=conf.cooling_time,
            log_path=conf.log_path,
            gpu=conf.gpu,
            name=conf.name,
        )
        node.stake(conf.stake)
        await self._try_spin_up(node, 1, max_retries)

    async def start_all_nodes(self):
        await asyncio.gather(*[
            self.start_node(conf, network)
            for conf in self.configs
            for network in conf.networks
        ])

    def start_nodes_as_subprocess(self):
        for conf in self.configs:
            for network in conf.networks:
                sync_start_node = lambda conf, net: asyncio.run(self.start_node(conf, net))
                proc = multiprocess.Process(target=sync_start_node, args=(conf,network))
                proc.start()
                self.process.append(proc)

    async def _run_nodes(self):
        multicoring = False if os.environ.get('SINGLE_CORE') else True
        # sync_spin_up = lambda conf: asyncio.run(self._spin_up(conf))
        if not multicoring:
            await self.start_all_nodes()
            return 
        self.start_nodes_as_subprocess()

    def run_nodes(self):
        print(f"[*] Building and running {len(self.configs)} nodes")
        try:
            asyncio.run(self._run_nodes())
        except KeyboardInterrupt:
            for proc in self.process:
                proc.kill()

    @classmethod
    def run_nodes_from_config_file(cls, path:str):
        with open(path, 'r') as file:
            if FileDetector.is_json(path):
                confs = json.loads(file.read())
            elif FileDetector.is_yaml(path):
                confs = yaml.safe_load(file)
            else:
                raise NotSupportedError(
                    f"Config file {path} is not supported. Please use one of the "
                    "following formats: [json, yaml]"
                )
        if isinstance(confs, dict):
            parsed = []
            for k, v in confs.items():
              v.update({'name': k})
              parsed.append(v)
            confs = parsed
            del parsed

        configs = [NodeConfig.from_dict(conf_dict) for conf_dict in confs]
        cls(configs).run_nodes()


def cli():
    """Testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--private-key', type=str)
    parser.add_argument('-s', '--stake', type=int)
    parser.add_argument('-b', '--bad-result', action=argparse.BooleanOptionalAction)
    parser.add_argument('-i', '--id', type=int)
    parser.add_argument('-e', '--env-file', type=str)
    parser.add_argument('-B', '--bad-chance', type=float)
    parser.add_argument('-r', '--rpc-url', type=str)
    parser.add_argument('-n', '--network', action='append')
    parser.add_argument('-a', '--api-key', type=str)
    parser.add_argument('-c', '--config', type=lambda s: s.split(','))
    parser.add_argument('--cooling-time', type=int)
    parser.add_argument('--anvil-config', type=str)
    
    args = parser.parse_args()

    if args.config:
        for js_path in args.config:
            Runner.run_nodes_from_config_file(js_path)
        return

    assert isinstance(args.stake, int)
    private_key = args.private_key if not args.private_key is None else None
    staking_amount = args.stake or 100
    anvil = None

    # It's valid just for anvil tests
    if args.env_file:
        dot = dotenv_values(args.env_file)
        private_key = dot["PRIVATE_KEY"]
        Global.update()

    if args.id and private_key is None:
        assert args.anvil_config or Global.anvil_config_path

        config_path = args.anvil_config or Global.anvil_config_path
        Global.anvil_config_path = config_path
        anvil = Anvil(config_path, build_envfile=True)
        account = getattr(anvil, f"node{args.id}")
        private_key = account.private_key

    if args.rpc_url:
        os.environ["FREEWILLAI_RPC"] = args.rpc_url
        Global.update()

    # At moment api_key is more important than rpc_url
    uri = args.api_key or args.rpc_url

    cooling_time = args.cooling_time or 0

    # If not uri build network by environment
    network = args.network or Network(uri)

    assert private_key

    max_retries = 5
    async def try_spin_up(num_retry: int):
        if max_retries < num_retry:
            return
        try:
            node = Node(
                private_key=private_key,
                test_bad_result=args.bad_result,
                test_bad_chance=args.bad_chance,
                anvil=anvil,
                network=network,
                cooling_time=cooling_time,
            )
            node.stake(staking_amount)
            await node.spin_up()

        except Fatal as err:
            raise Fatal(err)

        except Exception:
            err_path = f"/tmp/freewillai-error-in-{num_retry}"
            with open(err_path, "w") as err_file:
                err_file.write(traceback.format_exc(chain=False))
            print(f"[!] Retrying due to an unexpected error in iteration {num_retry}/{max_retries}")
            print(f"  > To see the complete log of the error=`cat {err_path}`\n")
            with open(err_path, 'r') as f:
                print(f.read())
            return await try_spin_up(num_retry+1)

    asyncio.run(try_spin_up(1))


def exit_handler(*_):
    print("[!] Node terminated by user")
    os._exit(1)


if __name__ == '__main__':
    # Capture SIGINT and SIGTERM
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    cli()
