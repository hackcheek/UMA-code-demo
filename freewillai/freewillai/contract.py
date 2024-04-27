import logging
import time
from typing import Any, Dict, Tuple, Optional, List, cast
from eth_account.signers.local import LocalAccount
from hexbytes import HexBytes
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.datastructures import AttributeDict
from web3.middleware import construct_sign_and_send_raw_middleware
from web3.types import TxParams, TxReceipt
from freewillai.doctypes import Abi, Bytecode
from freewillai.exceptions import UserRequirement, ContractError, Fatal
from freewillai.globals import Global
from freewillai.common import Anvil, ContractNodeResult, Network
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


# TODO: implement this class to use in the contract handled by network
class BlockchainInterface(ABC):
    """ChainProvider is a class to interact with the blockchain"""

    @abstractmethod
    def send_view_function(self, func_name: str, *args, log=True):
        """Send a view function to the blockchain"""

    @abstractmethod
    def send_transact_function(
        self,
        func_name: str,
        *args,
        log=True,
        max_retries: int,
        num_retry: int,
        add_nonce: int,
    ) -> HexBytes:
        """Send a transaction function to the blockchain"""

    @abstractmethod
    def wait_for_transaction(
        self,
        tx_hash: HexBytes,
        raise_on_error: bool
    ) -> TxReceipt:
        """Wait for a transaction to be mined"""


class Contract:
    def __init__(
        self, 
        account: LocalAccount,
        address: Optional[str] = None, 
        abi_path: Optional[str] = None, 
        abi: Optional[Abi] = None,
        bytecode: Optional[Bytecode] = None,
        constructor_args: Tuple[Any] = tuple(),
        network: Optional[Network] = None,
        allow_sign_and_send: bool = False,
        log = True,
    ):
        assert abi_path or abi
        assert bytecode or address
        
        self.abi = abi
        self.bytecode = bytecode
        self.address = address
        self.account = account

        # If network is None build by environment
        self.network = network or Global.network or Network.build()
        self.w3 = self.network.connect()

        if allow_sign_and_send:
            self.network.allow_sign_and_send(account)

        if (not allow_sign_and_send
            and self.network.middleware_onion.get('allow_sign_and_send')
        ):
            import sys
            logger.warn(f"[WARNING] Automatic sign and send is allowed in '{self.name}'")


        if self.address and self.w3.eth.get_code(self.address).hex() == '0x':
            raise Fatal(
                f"{self.name} not found with this address={self.address}"
            )

        if self.abi is None and not abi_path is None:
            with open(abi_path) as abifile:
                self.abi = abifile.read()

        assert self.w3.is_connected(), "Not connected to Ethereum node"

        if not bytecode is None:
            self.contract = self.w3.eth.contract(bytecode=self.bytecode, abi=self.abi)
            tx_hash = self.contract.constructor(*constructor_args).transact()
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            self.address = tx_receipt['contractAddress']

        if not self.address is None:
            self.address = Web3.to_checksum_address(self.address)
            self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)

    def name(self) -> str:
        return ""

    @property
    def eth_balance(self):
        return self.w3.eth.get_balance(self.account.address)

    def _get_params(self, add_nonce:int=0) -> Dict:
        # print(f"{int(self.w3.eth.gas_price * self.network.gas_multiplier)=}")
        params = {
            "chainId": self.w3.eth.chain_id,
            "from": self.account.address,
            "gasPrice": self.w3.eth.gas_price,
            "nonce": self.w3.eth.get_transaction_count(
                self.account.address
            ) + add_nonce
        }

        # gasPrice + 10% to address replacement transaction underpriced error
        if 'sepolia' in self.network.id:
            increased_gas_price = self.w3.eth.gas_price * 1.1
            params.update({"gasPrice": int(increased_gas_price)})
        
        if self.network.id == 'devnet/anvil':
            params.update({
                "gasPrice": 1875000000,
                "gas": 30000000,
            })

        return params


    def wait_for_transaction(self, tx_hash: HexBytes, raise_on_error: bool = True) -> TxReceipt:
        logger.info(f"\n"
            f'[*] Waiting for transaction...\n'
            f'  > transaction hash: {tx_hash.hex()}'
        )
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if raise_on_error and receipt['status'] == 0:
            self.check_transaction(tx_hash)
        logger.info(f'{tx_hash.hex()} mined')
        return receipt


    def send_transact_function(
        self,
        func_name: str,
        *args,
        log: bool = True,
        max_retries: int = 5,
        num_retry: int = 0,
        add_nonce: int = 0,
    ) -> HexBytes:
        logger.debug(f"{func_name=}, {args=}")
        contract_function = getattr(
            self.contract.functions, func_name
        )(*args)

        try:
            tx_params = self._get_params(add_nonce)

            def _send_tx(tx_params):
                transaction = contract_function.build_transaction(tx_params)
                signed_transaction = self.w3.eth.account.sign_transaction(
                    transaction, self.account.key.hex()
                )
                return self.w3.eth.send_raw_transaction(
                    signed_transaction.rawTransaction
                )

            try:
                tx_hash = _send_tx(tx_params)
            except TypeError as err:
                if 'gas' in err.args[0]:
                    tx_params.update({'gas': 2_000_000})
                    tx_hash = _send_tx(tx_params)
                else:
                    raise TypeError(err)
                

            if log:
                logger.info(f'\n'
                    f'[*] Executing transact function "{func_name}" from "{self.name}"\n'
                    f'  > transaction hash: {Web3.to_hex(tx_hash)}')

            return tx_hash

        except ValueError as err:
            if max_retries < num_retry:
                raise Fatal(f"Max retries raised err={err}")
            time.sleep(2)
            d = err.args[0]
            if d["message"] == 'replacement transaction underpriced': 
                logger.error(f"[!] {d['message']}, retrying with nonce +1") 
                return self.send_transact_function(
                    func_name, *args, log=log, num_retry=num_retry, add_nonce=1
                )

            elif d['message'] == 'nonce too low':
                logger.error(f"[!] {d['message']}, retrying with nonce +1") 
                return self.send_transact_function(
                    func_name, *args, log=log, num_retry=num_retry, add_nonce=1
                )

            elif d['message'] == 'transaction already imported':
                logger.error(f"[!] {d['message']}, retrying") 
                time.sleep(0.1)
                return self.send_transact_function(
                    func_name, *args, log=log, num_retry=num_retry, add_nonce=1
                )

            elif d['message'] == 'nonce too high':
                logger.error(f"[!] {d['message']}, retrying with nonce -1") 
                return self.send_transact_function(
                    func_name, *args, log=log, num_retry=num_retry, add_nonce=-1
                )

            elif d['code'] == -32000:
                num_retry += 1
                logger.error(f"Bypassing Error: {d['message']}")
                logger.error(f"  > Retrying {func_name} {num_retry}/{max_retries}")
                return self.send_transact_function(func_name, *args, log=log, num_retry=num_retry)

            raise ValueError(err)


    def send_view_function(self, func_name: str, *args, log=True):
        contract_function = getattr(
            self.contract.functions, func_name
        )(*args)
        if log:
            logger.info(f'[*] Executing view function "{func_name}" from "{self.name}"')

        logger.debug(f"{func_name=}, {args=}")
        return contract_function.call()


    def check_transaction(self, tx_hash):
        tx = cast(Dict, self.w3.eth.get_transaction(tx_hash))
        replay_tx: TxParams = {
            'to': tx['to'],
            'from': tx['from'],
            'value': tx['value'],
            'data': tx['input'],
        }
        try:
            self.w3.eth.call(replay_tx, tx['blockNumber'])
        except ContractLogicError as err:
            raise ContractError(err)


class TokenContract(Contract):
    def __init__(
        self, 
        account, 
        address: Optional[str] = None,
        abi_path: str = Global.token_abi_path,
        bytecode: Optional[Bytecode] = None, 
        abi: Optional[Abi] = None,
        network: Optional[Network] = None,
        allow_sign_and_send: bool = False,
    ):
        network = network or Network.build()
        address = address or network.token_address or Global.token_address
            
        super().__init__(
            account,
            abi=abi,
            bytecode=bytecode,
            address=address, 
            abi_path=abi_path, 
            network=network,
            allow_sign_and_send=allow_sign_and_send
        )

    @property
    def name(self) -> str:
        return "TokenContract"

    @property
    def fwai_balance(self):
        return self.get_balance_of(self.account.address)

    @property
    def owner(self) -> str:
        return self.send_view_function("owner")

    def approve(self, address, amount) -> HexBytes:
        return self.send_transact_function("approve", address, amount)

    def get_balance_of(self, address) -> float:
        return self.send_view_function("balanceOf", address)

    def initialize(self) -> HexBytes:
        return self.send_transact_function("initialize")

    def mint(self, address, amount) -> HexBytes:
        return self.send_transact_function("mint", address, amount)

    def burn(self, address, amount) -> HexBytes:
        return self.send_transact_function("burn", address, amount)


class TaskRunnerContract(Contract):
    def __init__(
        self, 
        account, 
        token_address: str = Global.token_address,
        address: Optional[str] = None,
        abi_path: str = Global.task_runner_abi_path,
        network: Optional[Network] = None,
        abi: Optional[Abi] = None,
        bytecode: Optional[Bytecode] = None, 
        token_contract: Optional[TokenContract] = None,
        allow_sign_and_send: bool = False,
    ):
        network = network or Network.build()
        address = address or network.task_runner_address or Global.task_runner_address
        if not token_contract:
            token_address = token_address or network.token_address or Global.token_address
        super().__init__(
            account,
            abi=abi,
            bytecode=bytecode,
            address=address, 
            abi_path=abi_path, 
            constructor_args=(token_address,),
            network=network,
            allow_sign_and_send=allow_sign_and_send,
        )
        self.token = token_contract or TokenContract(
            account, address=token_address, network=network
        )
        self.task_price = self.contract.functions.taskPrice().call()

        ## TODO: Decomment this after set public stakingMinimum in contract
        # self.staking_minimum = self.contract.functions.stakingMinimum().call()
        self.staking_minimum = 100

    @property
    def name(self) -> str:
        return "TaskRunnerContract"

    def check_if_ready_to_validate(self, task_id: int, log: bool = True):
        return self.send_view_function('checkIfReadyToValidate', task_id, log=log)

    def get_staking_amount(self, address, log=True):
        return self.send_view_function("stakingAmounts", address, log=False)

    def is_in_timeout(self, task_id: int, log: bool = True) -> int:
        return self.send_view_function('isInTimeout', task_id, log=log)

    def get_available_task_results(
        self, 
        task_id: int, 
        log: bool = True
    ) -> List[ContractNodeResult]:
        results: List[Tuple[str, str, int]] = self.send_view_function(
            'getAvailableTaskResults', task_id, log=log)
        return list(map(lambda tup: ContractNodeResult(*tup), results))

    def add_task(
        self, model_url: str,
        dataset_url: str,
        min_time: int = 1,
        max_time: int = 200,
        min_results: int = 2,
    ) -> Tuple[HexBytes, AttributeDict]:
        account_balance = self.token.get_balance_of(self.account.address)

        if account_balance < self.task_price: 
            raise UserRequirement(
                f'Not enough FWAI tokens to add task.\n'
                f"  > Your balance (FWAI): {account_balance}\n"
                f"  > Price of task: {self.task_price}\n"
                f'Please buy {self.task_price - account_balance} FWAI as minimum'
            )

        t1 = time.perf_counter()
        tx_hash = self.token.approve(self.address, self.task_price)
        receipt = self.wait_for_transaction(tx_hash, True)
        t2 = time.perf_counter()

        tx_hash = self.send_transact_function(
            'addTask', model_url, dataset_url, min_time, max_time, min_results
        )
        receipt = self.wait_for_transaction(tx_hash, True)
        t3 = time.perf_counter()

        # TODO: develop a timer handler as an option to select
        # print(f"[TIMER] approve took {t2-t1:.3f}")
        # print(f"[TIMER] add_task took {t3-t2:.3f}")

        logging.debug(f'{tx_hash=}, {self.task_price=}')
        
        try:
            event_data = self.contract.events.TaskAdded().process_receipt(receipt)[0]
        except IndexError as err:
            raise IndexError(err)

        logger.info(f"\n"
            f"[*] Task added\n"
            f"  > model_url: {model_url}\n"
            f"  > dataset_url: {dataset_url}\n"
            f"  > Transaction hash: {Web3.to_hex(tx_hash)}"
        )
        return tx_hash, event_data

    def get_avaialble_task_results_count(self, task_id) -> int:
        return self.send_view_function('getAvailableTaskResultsCount', task_id)

    def validate_task_if_ready(self, task_id: int) -> HexBytes:
        return self.send_transact_function('validateTaskIfReady', task_id)

    def check_staking_enough(self):
        return self.send_view_function("checkStakingEnough")

    def is_validated(self, task_id, log=True) -> bool:
        return self.send_view_function("isValidated", task_id, log=log)

    def get_tasks_count(self) -> int:
        available_tasks_count = self.send_view_function('getAvailableTasksCount')
        logging.info(f'[*] Available tasks count = {available_tasks_count}')
        return available_tasks_count

    def get_task_result(self, model_url, dataset_url) -> str:
        return self.send_view_function('getTaskResult', model_url, dataset_url)

    def get_task_time_left(self, task_id) -> int:
        return self.send_view_function('getTaskTimeLeft', task_id)

    def submit_result(self, task) -> HexBytes:
        tx_hash = self.send_transact_function(
            'submitResult', 
            task.id, task.model_url, task.dataset_url, task.result_url
        )
        logger.info(task)
        logger.info(f"\n"
            f"[*] Result Submitting\n"
            f"  > Transaction hash: {Web3.to_hex(tx_hash)}"
        )
        return tx_hash

    def generate_block(self):
        """Testing func"""
        self.send_transact_function('validateTaskIfReady', 0, log=False)

    def stake(self, amount) -> HexBytes:
        return self.send_transact_function("stake", amount)

    def unstake(self, amount) -> HexBytes:
        return self.send_transact_function("unstake", amount)
