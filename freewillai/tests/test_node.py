import pytest
import sys
import os
import logging
import asyncio
import subprocess
import numpy as np
from typing import Optional, List, Dict
from numpy import ndarray

from freewillai.common import IPFSBucket, FWAIModel, FWAIDataset, NonTuringModel
from freewillai.node import Node
from freewillai.utils import get_url, load_global_env


class Global_test: 
    csv_path = 'bucket/test/datasets/keras_testing_dataset.csv'
    image_path = 'bucket/test/datasets/cat.png'
    numpy_path = 'bucket/test/datasets/cat_img_pytorch.npy'
    model_path = 'bucket/test/models/test_model.onnx'

    dataset_hashes: Optional[List] = None
    model_hash: Optional[str] = None
    
    hashes = dict(
        Keras_onnx_model = 'Qmd8zXu5Z8QZ7RYxnzjSW4zquwzu5M155o3mXV8MB4MQGi',
        dataset_keras_csv = 'QmPodbGqLsvbqczgMWarztX6tqhGMQiYEfUQPM7ph2tu2Q',
        sklearn_onnx_model = 'QmT3j1Lo5WCFWQrDdtHiiEetwADoNA7cvU8S97p65CYAN3',
        dataset_sklearn_csv = 'QmTgXsSDajjTzv3Qx7FH5d5TRFbzVNEwVeCcACnsc1L5cX',
        pytorch_onnx_model = 'QmTFTNdrcYYErtjA9hNysZv8VeneCvdLLxGWkRhEdfER7V',
        Image_path_pytorch = 'QmVQ8p73epScpKknQzW4VkSbgq1sFdZhfRFuZ6cqibU2BU',
        pil_image_pytorch = 'QmbfDdE1kHHSJFEJ7eySKv9UpzqmHxSRiPvyhjpqoiD8Bf',
        image_pytorch_tensor = 'QmeUKjDGzHzU2dLvrMfWaap9nVxRJQvSDVuiRQcxBa9wy7',
    )

    nodes: Dict[int, str] = {}
    for num in (1,2):
        env_path = f"worker{num}.env"
        loaded = load_global_env(env_path)
        if not loaded:
            raise RuntimeError(f"Path `{env_path}` not found")
        nodes[num] = os.environ['PRIVATE_KEY']
    


class TestIPFS:
    # WARNING: DO NOT ADD VARIABLES HERE. 
    # PYTEST WILL RESET THEM BEFORE RUNNING THE NEXT TEST
    def test_upload_dataset(self):
        async def _upload_files():
            tasks = [
                IPFSBucket.upload(Global_test.csv_path, file_type='dataset'),
                IPFSBucket.upload(Global_test.image_path, file_type='dataset'),
                IPFSBucket.upload(Global_test.numpy_path, file_type='dataset')]
            Global_test.dataset_hashes = [
                ipfs.cid for ipfs in await asyncio.gather(*tasks)
            ]
        asyncio.run(_upload_files())

    def test_upload_model(self):
        async def _upload_files():
            ipfs_model = await IPFSBucket.upload(Global_test.model_path, file_type='model')
            Global_test.model_hash = ipfs_model.cid
        asyncio.run(_upload_files())

    def test_download_dataset(self):
        assert Global_test.dataset_hashes is not None
        async def _download_files():
            tasks = [
                IPFSBucket.download(hash, file_type='dataset')
                for hash in Global_test.dataset_hashes
            ]
            await asyncio.gather(*tasks)
        asyncio.run(_download_files())

    def test_download_model(self):
        assert Global_test.model_hash is not None
        async def _download_files():
            task = await IPFSBucket.download(Global_test.model_hash, file_type='model')
        asyncio.run(_download_files())

    def test_download_all(self):
        logging.warning(Global_test.hashes.values())
        async def _download_files():
            tasks = [
                IPFSBucket.download(hsh)
                for hsh in Global_test.hashes.values()
            ]
            await asyncio.gather(*tasks)

        asyncio.run(_download_files())


class TestInference:

    def test_sklearn_csv(self):
        model_url = get_url(Global_test.hashes['sklearn_onnx_model'])
        dataset_url = get_url(Global_test.hashes['dataset_sklearn_csv'])
        wanted = np.array([1,1,1,0,0,0])

        model: NonTuringModel = FWAIModel.load(model_url)
        dataset: ndarray = FWAIDataset.load(dataset_url)
        result: ndarray = model.inference(dataset)

        assert (result == wanted).all()


def ipfs_anvil_setup():
    # Run ipfs
    subprocess.call("../scripts/run_ipfs.sh", shell=True)

    # Run anvil
    subprocess.call("../scripts/run_anvil.sh", shell=True)

    # Deploy contracts and set balance in eth
    subprocess.call("../scripts/setup_anvil.sh", shell=True)

    # Mint to nodes and client
    subprocess.call("../scripts/mint_fwai.sh devnet/anvil", shell=True)
    

class TestAsyncNodes:
    def test_two_nodes(self):
        network = "devnet/anvil"
        amount_to_stake = 500

        # Intance nodes
        node1 = Node(private_key=Global_test.nodes[1], network=network)
        node2 = Node(private_key=Global_test.nodes[2], network=network)

        # Staking
        node1.stake(amount_to_stake)
        node2.stake(amount_to_stake)

        # Run nodes async
        async def run_nodes():
            await asyncio.gather(node1.spin_up(), node2.spin_up())
        sys.exit(asyncio.run(run_nodes()))
