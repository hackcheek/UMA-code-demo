"""
Tests all functions of freewillai
Develop tests with functions in freewillai/__init__.py
"""
import freewillai
import asyncio


class TestUploadAndDownload:
    """
    TODO: Develop tests with all kind of models and datasets
    """
    def test_upload_keras_model(self):
        model = 'bucket/test/models/keras_model_dnn'
        asyncio.run(freewillai.upload_model(model, input_format='csv'))

    def test_upload_image_dataset(self):
        image = '../bucket/test/datasets/cat.png'
        asyncio.run(freewillai.upload_dataset(image))


class TestRunTask:
    """
    TODO: Develop tests with all kind of models and datasets
    """
    ...
