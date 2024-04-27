"""
Pipelines built in Dagger to run CI/CD
"""
import asyncio
import json
import os
import re
import sys
import dagger
import docker
import socket
from typing import Dict, List, Literal, Optional, Union, cast
from dotenv import load_dotenv


class DemoPipeline:
    def __init__(
        self,
        config:dagger.Config,
        dest_dir:str="containers/",
        envfile:str="dagger.env",
        docker_socket_path:str="/var/run/docker.sock",
        docker_from_env:bool=False,
    ):
        """
        Pipeline to deploy demo on playground.freewillai.org
        
        Arguments
        ---------

        Usage
        -----

        Use deploy() to start up docker containers necessaries for demo
        >>> config = dagger.Config()
        >>> DemoPipeline.deploy(config)
        
        Use run_<name>() to run just some docker containers.
        In this case run anvil and ipfs
        >>> config = dagger.Config()
        >>> async with dagger.Connection(config) as client:
        ...     pipe = DemoPipeline(client)
        ...     await pipe.run_anvil()
        ...     await pipe.run_ipfs()
        ...
        ...     # In simultaneously
        ...     await asyncio.gather(pipe.run_anvil(), pipe.run_ipfs())
        """

        if not os.path.exists(docker_socket_path):
            raise ConnectionError("Docker is not running. Please start it up")

        self.config = config
        self.dest_dir = dest_dir
        self.envfile = envfile
        self.docker_socket_path = docker_socket_path
        self.docker_socket = "unix:/" + docker_socket_path

        self.docker_client = (
            docker.from_env() if docker_from_env
            else docker.DockerClient(base_url=self.docker_socket)
        )

        self.anvil_tarpath = os.path.join(self.dest_dir, "anvil.dagger.tar")
        self.ipfs_tarpath = os.path.join(self.dest_dir, "ipfs.dagger.tar")
        self.node_tarpath = os.path.join(self.dest_dir, "node.dagger.tar")
        self.repl_tarpath = os.path.join(self.dest_dir, "repl.dagger.tar")

        # Containers name
        self.anvil_name = "freewillai_anvil"
        self.ipfs_name = "freewillai_ipfs"
        self.repl_name = "freewillai_repl"
        self.workers_preffix = "freewillai_worker"

        self.anvil_port = 8545
        self.repl_port = None

        self.token_address = None
        self.task_runner_address = None

        # Always the first private key in private_keys is owner
        self.private_keys: List[str] = []
        self.private_keys_path: Union[None, str] = None

        # Running containers
        self.building_containers: Dict[str, dagger.Container] = {}
        self.running_containers = {}


        nets = [net.name for net in self.docker_client.networks.list()]
        if "main" in nets:
            self.main_network = self.docker_client.networks.get("main")
        else:
            self.main_network = self.docker_client.networks.create(
                "main", driver="bridge"
            )

        loaded = load_dotenv(".env")
        self.owner_pk = os.getenv("PRIVATE_KEY") 
        if not loaded or self.owner_pk is None:
            raise RuntimeError(
                "Are you the Admin of the server? It raises an notification to the admin"
            )


    async def __aenter__(self):
        self.conn = dagger.Connection(self.config)
        self.client = await self.conn.__aenter__()
        self.host = self.client.host()

        # App directory
        self.app_dir = self.host.directory(
            ".",
            include=(
                "scripts/",
                "freewillai/",
                "contracts/",
                "lib/",
                "requirements.txt",
                "foundry.toml",
                ".env"
            )
        )
        print(f"{self.docker_client.containers.list(all=True)=}")
        return self


    async def __aexit__(self, *args, **kwargs):
        self.cleanup()
        return await self.conn.__aexit__(*args, **kwargs)

    def get_worker_name(self, id:int):
        return self.workers_preffix + str(id)
        

    def is_valid_private_key(self, key: str) -> bool:
        # Ethereum private keys are 64 hex characters long, not counting the '0x' prefix.
        if not isinstance(key, str):
            return False

        if key.startswith('0x'):
            key = key[2:]

        # Check if the key is 64 characters long and if it's a valid hex number
        return len(key) == 64 and all(c in '0123456789abcdefABCDEF' for c in key)


    def get_private_keys(self):
        if self.private_keys_path is None:
            raise RuntimeError(f"Please first run build_workers")
        with open(self.private_keys_path, "r") as file:
            lines = file.readlines()
        for line in lines:
            pk = line.strip()
            if not self.is_valid_private_key(pk):
                raise RuntimeError(f"{self.private_keys_path=} is not valid")
            self.private_keys.append(pk)
        return self.private_keys


    def base_python_container(
        self,
        client:Optional[dagger.Client]=None
    ) -> dagger.Container:
        requirements_file = self.client.host().file("./requirements.txt")
        client = client or self.client
        return (
            client.container()
            .from_("python:3.10-slim")
            .with_file("./requirements.txt", requirements_file)
            .with_exec(("apt-get", "upgrade", "-y"))
            .with_exec(("apt-get", "update"))
            .with_exec(("apt-get", "install", "git", "cmake", "curl", "gcc", "-y"))
            .with_exec(("apt", "clean"))
            .with_exec(("sh", "-c", "curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"))
            .with_env_variable("PATH", "/root/.cargo/bin:${PATH}", expand=True)
            .with_exec(("pip", "install", "--upgrade", "pip"))
            .with_exec(("pip", "install", "--no-cache-dir", "-r", "requirements.txt"))
            .with_exec(("pip", "install", "git+https://github.com/hackcheek/tensorflow-onnx"))
            .with_default_args()
        )


    def _get_set_balance_cmd(self, account, value:int, endpoint:str):
            data = json.dumps({
                "method": "anvil_setBalance",
                "params": [account.address, value],
                "id": 1,
                "jsonrpc": "2.0"
            })
            return (
                "curl", endpoint,
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "--data", repr(data)
            )


    def _get_image_from_tarfile(self, path:str):
        with open(path, 'rb') as tarfile:
            images = self.docker_client.images.load(tarfile.read())
        return images[0]


    async def build_anvil(
        self,
        port:int=8545,
        export_path:Optional[str]=None
    ) -> dagger.Container:
        if os.path.exists(self.anvil_tarpath):
            os.remove(self.anvil_tarpath)

        if export_path:
            self.anvil_tarpath = export_path

        anvil: dagger.Container = (
            self.client.pipeline("anvil")
            .container()

            # # For testing in macos
            # .container(platform=dagger.Platform("linux/arm64"))

            .from_("ghcr.io/foundry-rs/foundry")
            .with_directory("/app", self.app_dir)
            .with_exec(("mkdir /anvil"))
            .with_exec("apk add curl")
            .with_workdir('/app/scripts/dagger')
            .with_entrypoint((f"./run_anvil.sh", f"{str(port)}"))
        )

        self.building_containers["anvil"] = anvil
        self.anvil_port = port

        exported = await anvil.export(self.anvil_tarpath)
        if not exported:
            raise RuntimeError("Anvil image has not been exported")

        return anvil


    async def build_ipfs(
        self,
        api_port:int=5001,
        gateway_port:int=8080,
        export_path:Optional[str]=None,
    ) -> dagger.Container:
        self.ipfs_api_port = api_port
        self.ipfs_gateway_port= gateway_port

        if os.path.exists(self.ipfs_tarpath):
            os.remove(self.ipfs_tarpath)

        if export_path:
            self.ipfs_tarpath = export_path

        ipfs: dagger.Container = (
            self.client.container()
            .from_("ipfs/kubo:latest")
            .with_file(
                "/run_ipfs.sh",
                self.app_dir.file("scripts/dagger/run_ipfs.sh")
            )
            .with_exposed_port(api_port)
            .with_exposed_port(gateway_port)
            .with_default_args()
            .with_entrypoint((
                "./run_ipfs.sh", str(api_port), str(gateway_port)
            ))
        )

        self.building_containers["ipfs"] = ipfs

        exported = await ipfs.export(self.ipfs_tarpath)
        if not exported:
            raise RuntimeError("IPFS image has not been exported")

        return ipfs


    async def build_workers(
        self,
        stake:int=500,
        cooling_time:float=0.1,
        num_of_workers:int=1,
    ):
        print("[*] Building containers of nodes")
        anvil_rpc = f'http://anvil:{self.anvil_port}'

        anvil_container = self.get_anvil_container()
        if not anvil_container:
            raise RuntimeError("Run run_anvil method before build_workers")

        token_address = self.token_address or self.get_token_address(anvil_container)
        if token_address is None:
            raise RuntimeError("Run run_anvil method before build_workers")

        creator = (
            self.base_python_container()
            .with_directory("/app", self.app_dir)
            .with_workdir("/app")
            .with_entrypoint((
                'python', '-m', 'scripts.dagger.run_nodes',
                '--anvil-rpc', anvil_rpc,
                '--token-address', token_address,
                '--stake', str(stake),
                '--cooling-time', str(cooling_time),
                '--owner-pk', cast(str, self.owner_pk),
                '--num-of-workers', str(num_of_workers),
            ))
        )

        self.building_containers["node"] = creator
        exported = await creator.export(self.node_tarpath)
        if not exported:
            raise RuntimeError("Node image has not been exported")


    async def build_repl(self, port:int=5555):
        print("[*] Building containers of nodes")
        self.repl_port = port
        anvil_rpc = f'http://anvil:{self.anvil_port}'
        try:
            anvil_container = self.get_anvil_container()
            self.token_address = self.token_address or self.get_token_address(anvil_container)
        except:
            if not os.path.exists('anvil.env'):
                raise RuntimeError("Run run_anvil method before build_repl")

        if self.token_address is None:
            load_dotenv("anvil.env")
            if os.environ.get('FREEWILLAI_TOKEN_ADDRESS') is None:
                raise RuntimeError("Run run_anvil method before build_workers")
            self.token_address = os.environ.get('FREEWILLAI_TOKEN_ADDRESS')
        
        repl_dir = self.host.directory("./web/demo")
        fwai_dir = self.host.directory("./freewillai")
        contracts_dir = self.host.directory("./contracts/")
        demos_dir = self.host.directory("./demos")
        dotenv_file = self.host.file(".env")
        
        creator = (
            self.base_python_container()
            .with_exec((
                "pip", "install",
                "fastapi", "uvicorn", "nest_asyncio", "RestrictedPython"
            ))
            .with_exposed_port(port)
            .with_directory("/repl", repl_dir)
            .with_directory("/repl/freewillai", fwai_dir)
            .with_directory("/repl/contracts", contracts_dir)
            .with_directory("/repl/demos", demos_dir)
            .with_file("/repl/.env", dotenv_file)
            .with_workdir("/repl")
            .with_entrypoint((
                "python", "-m", "app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--anvil-rpc", anvil_rpc,
            ))
        )

        self.building_containers["repl"] = creator
        exported = await creator.export(self.repl_tarpath)
        if not exported:
            raise RuntimeError("Repl image has not been exported")


    def get_anvil_container(self):
        return self.docker_client.containers.get(self.anvil_name)

    def get_ipfs_container(self):
        return self.docker_client.containers.get(self.ipfs_name)

    def get_repl_container(self):
        return self.docker_client.containers.get(self.repl_name)

    def get_worker_container(self, id):
        return self.docker_client.containers.get(self.get_worker_name(id))

    def is_anvil_running(self) -> bool:
        anvil = self.get_anvil_container()
        if not anvil:
            return False
        return True


    def get_ip_from_network(
        self,
        container_name:str,
        network_name:Optional[str]=None
    ):
        network_name = network_name or self.main_network.name
        try:
            network = self.docker_client.networks.get(network_name)
            
            for container in network.containers:
                if container.name == container_name:
                    return (
                        container.attrs
                        ['NetworkSettings']['Networks'][network_name]['IPAddress']
                    )
            raise RuntimeError(
                f"Container={container_name} is not in Network={network_name}"
            ) 

        except docker.errors.NotFound:
            print(f"Network '{network_name}' not found.")  


    def run_container(self, name:str, image, *args, **kwargs):
        try:
            container = self.docker_client.containers.get(name)
            container.remove(force=True)
        except docker.errors.NotFound:
            ...

        kw = dict(
            stdout=True, stderr=True, detach=True, remove=True,
            name=name, # network="host",
        )
        kw.update(kwargs)
        c = self.docker_client.containers.run(image, *args, **kw)
        self.main_network.connect(c)
        self.running_containers.update({name: c})
        return c


    def get_contract_address(
        self,
        anvil_container,
        contract: Literal['token', 'task_runner']
    ) -> str | None:
        log_string = anvil_container.logs().decode("utf-8")
        pattern = {
            'token': r'TokenAddress:([0-9a-fA-FxX]{42})',
            'task_runner': r'TaskRunnerAddress:([0-9a-fA-FxX]{42})'
        }
        match_ = re.search(pattern[contract], log_string)
        if match_ is None:
            return
        return match_.group(1)
    
    def get_token_address(self, anvil_container):
        return self.get_contract_address(anvil_container, 'token')

    def get_task_runner_address(self, anvil_container):
        return self.get_contract_address(anvil_container, 'task_runner')

    
    async def attach(self, log_path:str, read_time:float=0.2):
        with open(log_path, "r") as logfile:
            logfile.seek(0, 2)
            while True:
                lines = logfile.readlines()
                if not lines:
                    await asyncio.sleep(read_time)
                    continue

                for line in lines:
                    print(line)


    async def run_anvil(self, force_build:bool=False):
        print("[*] Spinning up anvil")
        if (
            not os.path.exists(self.anvil_tarpath)
            or force_build
        ):
            await self.build_anvil()

        image = self._get_image_from_tarfile(self.anvil_tarpath)
        container = self.run_container(
            self.anvil_name, image,
            # network=self.main_network.name,
            # network='host',
            ports={f'{self.anvil_port}/tcp': self.anvil_port},
        )

        self.task_runner_address = None
        while self.task_runner_address is None:
            await asyncio.sleep(1)
            self.token_address = self.get_token_address(container)
            self.task_runner_address = self.get_task_runner_address(container)
            print(f"[DEBUG] STDOUT: {container.logs().decode('utf-8')}")
            print(
                f"[DEBUG] Waiting for token_address={self.token_address} "
                f"and task_runner_address={self.task_runner_address}"
            )
        return container


    async def run_ipfs(self, force_build:bool=True, wait=True):
        print("[*] Starting IPFS daemon")
        if not os.path.exists(self.ipfs_tarpath) or force_build:
            await self.build_ipfs()
        image = self._get_image_from_tarfile(self.ipfs_tarpath)

        self.ipfs_api_url = f"http://0.0.0.0:{self.ipfs_api_port}"
        self.ipfs_gateway_url = f"http://0.0.0.0:{self.ipfs_gateway_port}"

        container = self.run_container(
            self.ipfs_name, image,
            # network=self.main_network.name,
            # network='host',
            ports={
                f'{self.ipfs_api_port}/tcp': self.ipfs_api_port,
                f'{self.ipfs_gateway_port}/tcp': self.ipfs_gateway_port,
            },
        )
        while wait:
            await asyncio.sleep(1)
            if 'Daemon is ready' in container.logs().decode('utf-8'):
                print(f"[DEBUG] IPFS daemon is ready")
                break
            
        return container


    def get_worker_log_dir(self, id, log_dir:str):
        if os.path.exists(log_dir.format(id)):
            return self.get_worker_log_dir(id+1, log_dir)
        if not os.path.exists(log_dir.format(id)):
            os.mkdir(log_dir.format(id))
        return log_dir.format(id)


    async def run_worker(self, id:int, force_build:bool=False, attach:bool=False):
        print(f"[*] Starting Worker {id}")
        if not os.path.exists(self.node_tarpath) or force_build:
            await self.build_workers()
        image = self._get_image_from_tarfile(self.node_tarpath)
        name = self.get_worker_name(id)
        log_path = os.path.abspath(os.path.join(self.dest_dir, "logs"))

        container = self.run_container(
            name, image,
            # network=self.main_network.name,
            # network='host',
            links={
                self.get_anvil_container().name: "anvil",
                self.get_ipfs_container().name: "ipfs"
            },
            volumes={log_path: {"bind": "/logs", "mode":"rw"}},
        )

        _log_dir = os.path.join(self.dest_dir, "/logs/fwai-workers-log-{}/")
        log_dir = self.get_worker_log_dir(id, log_dir=_log_dir)
        if attach:
            await self.attach(os.path.join(log_dir, f"worker{id}"))


    async def run_repl(self, force_build:bool=False, attach:bool=False):
        print("[*] Starting REPL")
        if not os.path.exists(self.repl_tarpath) or force_build:
            await self.build_repl()
        image = self._get_image_from_tarfile(self.repl_tarpath)
        log_path = os.path.abspath(os.path.join(self.dest_dir, "logs"))

        container = self.run_container(
            self.repl_name, image,
            links={
                self.get_anvil_container().name: "anvil",
                self.get_ipfs_container().name: "ipfs"
            },
            ports={
                f'{self.repl_port}/tcp': self.repl_port,
            },
            volumes={log_path: {"bind": "/logs", "mode":"rw"}},
        )
        if attach:
            await self.attach("/logs/std.txt")

    def cleanup(self):
        # Remove all stopped containers
        for container in self.docker_client.containers.list(
            all=True, filters={"status": "exited"}
        ):
            container.remove()

        # Remove all unused images (both dangling and unreferenced)
        self.docker_client.images.prune(filters={"dangling": False})

        # Remove all unused networks
        self.docker_client.networks.prune()

        # Remove all unused volumes
        self.docker_client.volumes.prune()

        print("Cleanup completed!")


    @classmethod
    async def deploy(
        cls,
        config:dagger.Config,
        num_of_workers:int,
        cooling_time:float=0.1,
        stake:int=1000,
        dest_dir:str="containers/"
    ):
        async with cls(config, dest_dir=dest_dir) as pipe:
            # Build and Run anvil and ipfs
            await asyncio.gather(
                pipe.run_anvil(force_build=True),
                pipe.run_ipfs(force_build=True)
            )

            # Build workers and client
            await asyncio.gather(
                pipe.build_repl(),
                pipe.build_workers(
                    stake=stake,
                    cooling_time=cooling_time,
                    num_of_workers=num_of_workers
                )
            )

            # Run workers and repl
            workers = (pipe.run_worker(id) for id in range(num_of_workers))
            await asyncio.gather(*workers, pipe.run_repl())


if __name__ == "__main__":
    config = dagger.Config(log_output=sys.stderr)
    async def test():
        await DemoPipeline.deploy(config, 3)
        print("FINISH")
    asyncio.run(test())
