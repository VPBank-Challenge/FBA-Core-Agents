import boto3
from core.embedding import BaseEmbedder

class AWSEmbedder(BaseEmbedder):
    def __init__(self, endpoint_name):
        self.endpoint = endpoint_name
        self.client = boto3.client('sagemaker-runtime')

    def embed(self, text):
        response = self.client.invoke_endpoint(...)
        return parsed_vector