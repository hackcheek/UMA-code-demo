import requests
import pandas as pd

from tempfile import NamedTemporaryFile


API_URL = "http://0.0.0.0:9595"
TEST_DATA_PATH = 'datasets/diamonds/new_records.csv'


class TestAPIEndpoints:
    def test_predict_upload_file_1(self):
        """
        Upload csv and return json
        """
        body = {
            'output_format': 'json'
        }
        files = {'data': open(TEST_DATA_PATH, 'rb')}
        resp = requests.post(API_URL + '/csv/predict', files=files, data=body)
        assert resp.status_code == 200

        json_response = resp.json()

        assert isinstance(json_response, dict)
        assert isinstance(json_response['result'], dict)
        assert json_response['output_format'] == 'json'


    def test_predict_1(self):
        """
        input json and return json
        """
        json_data = pd.read_csv(TEST_DATA_PATH).to_dict()
        body = {
            'data': json_data,
            'output_format': 'json'
        }
        resp = requests.post(API_URL + '/predict', json=body)
        assert resp.status_code == 200
        json_response = resp.json()

        assert isinstance(json_response, dict)
        assert isinstance(json_response['result'], dict)
        assert json_response['output_format'] == 'json'


    def test_predict_upload_file_2(self):
        """
        Upload csv and return csv
        """
        body = {
            'output_format': 'csv'
        }
        files = {'data': open(TEST_DATA_PATH, 'rb')}
        resp = requests.post(API_URL + '/csv/predict', files=files, data=body)
        assert resp.status_code == 200
        json_response = resp.json()

        assert isinstance(json_response, dict)
        assert isinstance(json_response['result'], str)
        assert json_response['output_format'] == 'csv'


    def test_predict_2(self):
        """
        input json and return csv
        """
        json_data = pd.read_csv(TEST_DATA_PATH).to_dict()
        body = {
            'data': json_data,
            'output_format': 'csv'
        }
        resp = requests.post(API_URL + '/predict', json=body)
        assert resp.status_code == 200
        json_response = resp.json()

        assert isinstance(json_response, dict)
        assert isinstance(json_response['result'], str)
        assert json_response['output_format'] == 'csv'


    def test_predict_3(self):
        """
        input csv string and return csv
        """
        with open(TEST_DATA_PATH, 'r') as f:
            content = f.read()
        body = {
            'data': content,
            'output_format': 'csv'
        }
        resp = requests.post(API_URL + '/predict', json=body)
        assert resp.status_code == 200
        json_response = resp.json()

        assert isinstance(json_response, dict)
        assert isinstance(json_response['result'], str)
        assert json_response['output_format'] == 'csv'


    def test_predict_4(self):
        """
        input csv string and return json
        """
        with open(TEST_DATA_PATH, 'r') as f:
            content = f.read()
        body = {
            'data': content,
            'output_format': 'json'
        }
        resp = requests.post(API_URL + '/predict', json=body)
        assert resp.status_code == 200
        json_response = resp.json()

        assert isinstance(json_response, dict)
        assert isinstance(json_response['result'], dict)
        assert json_response['output_format'] == 'json'
