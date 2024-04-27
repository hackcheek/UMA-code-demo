from aioipfs import aiohttp
import json
import os
import freewillai
import uvicorn
import docker
import time
import asyncio
import traceback
import nest_asyncio
from typing import Literal, Tuple
from fastapi import FastAPI, WebSocket, Request
from fastapi.logger import logger
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from freewillai.constants import TASK_RUNNER_CONTRACT_ABI_PATH, TOKEN_CONTRACT_ABI_PATH
from freewillai.globals import Global
# from web.demo.user import execute_user_code_async
from freewillai.contract import TaskRunnerContract, TokenContract
from freewillai.common import Network, SepoliaNetwork
from freewillai.utils import get_account, load_global_env
from web3 import Web3
from dataclasses import dataclass, field
from web.demo.db import NotionDB, Record
from concurrent.futures import ProcessPoolExecutor


IMAGE_NAME = 'py_executor'

host = "host"
anvil_endpoint = f'http://{host}:8545'
ipfs_host = host


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://playground.freewillai.org",
        "http://playground.freewillai.org",
        "https://demo.freewillai.org",
        "http://demo.freewillai.org",
        "https://freewillai.org",
        "https://freewillai.webflow.io",
        "http://freewillai.webflow.io",

        # For development
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = NotionDB()

app.mount("/css", StaticFiles(directory="web/demo/css"), name="css")
templates = Jinja2Templates(directory='web/demo/templates')


def parse_code(code):
    """TODO: parse the code for vulnerabilities"""
    ...


async def aexec(code):
    # Make an async function with the code and `exec` it
    exec(
        f'async def __FWAI_EXECUTION(): ' +
        ''.join(f'\n {l}' for l in code.split('\n'))
    )

    # Get `__ex` from local variables, call it and return the result
    return await locals()['__FWAI_EXECUTION']()


class DockerLogger:
    def __init__(self, container_name: str, client = None):
        self.container_name = container_name
        self.client = client or docker.from_env()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0.2)
        container = self.client.containers.get(self.container_name)
        if container.status == 'exited':
            container.remove(force=True)
            raise StopAsyncIteration
        return container.logs().decode('utf-8').strip()


async def run_docker(code, websocket):
    filename = '/tmp/code.py'
    dest = "saracatunga"

    with open(filename, 'w') as file:
        file.write(code)

    client = docker.from_env()
    container = client.containers.run(
        IMAGE_NAME, dest, detach=True, volumes=[f'{filename}:/repl/{dest}'],
        extra_hosts={"host": "0.0.0.0"}, network='host', stderr=True
    )
    logger = DockerLogger(container.name, client)
    async for logs in logger:
        await websocket.send_text(logs)


async def run_code(response, websocket):
    nest_asyncio.apply()

    custom_rpc = f'"{anvil_endpoint}",' if response.network == 'devnet/anvil' else ''
   
    code =  'import freewillai, warnings\n'
    code += 'warnings.filterwarnings("ignore")\n'
    code += f'freewillai.connect("{response.network}",{custom_rpc} ipfs_host="{ipfs_host}")\n'
    code += response.code

    try:
        if 'import logging' in str(code):
            raise RuntimeError('logging module is not supported use print instead')
        await run_docker(code, websocket)

    except Exception:
        output = traceback.format_exc()
        await websocket.send_text(output)


@app.websocket('/run_code')
async def run_code_demo(websocket: WebSocket):
    nest_asyncio.apply()
    await websocket.accept()
    record = Record(**await websocket.receive_json())
    await db.add_record(record)
    await run_code(record, websocket)


@app.get('/', response_class=HTMLResponse)
async def test(request: Request):
    return templates.TemplateResponse('test.html', {'request': request})


@dataclass
class InfoFilter:
    chains: list[str] = field(default_factory=list)


def load_json(path: str | os.PathLike):
    with open(path, 'r') as f:
        return json.load(f)

async def get_info(info_filter: InfoFilter | None = None):
    info_filter = InfoFilter()
    chains_info_path = '/tmp/updated_chain_info.json'
    Network.dump_networks(chains_info_path)
    chains_info = load_json(chains_info_path)
    if info_filter.chains:
        chains_info = {
            id: value
            for id, value in chains_info.items()
            if id in info_filter.chains
        }
    return json.dumps(dict(
        task_runner_abi = load_json(TASK_RUNNER_CONTRACT_ABI_PATH),
        token_abi = load_json(TOKEN_CONTRACT_ABI_PATH),
        chains = chains_info
        ))

@app.get('/info', response_class=JSONResponse)
async def info():
    return await get_info()

@app.post('/info', response_class=JSONResponse)
async def post_info(info_filter: InfoFilter):
    print("INFO_FILTER", info_filter)
    return await get_info(info_filter)


def cli():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--anvil-rpc", type=str)
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    Global.ipfs_host = 'ipfs'
    Global.ipfs_port = 5001  # HARDCODING
    os.environ["IPFS_HOST"] = 'ipfs'
    os.environ["IPFS_PORT"] = '5001'

    global anvil_endpoint
    anvil_endpoint = args.anvil_rpc
    uvicorn.run('web.demo.app:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    cli()
