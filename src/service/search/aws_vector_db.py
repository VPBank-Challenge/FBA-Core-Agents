class AWSVectorDBSearch(BaseSearchEngine):
    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def search(self, query, top_k=3):
        list_of_results = ["None because this is a placeholder for AWS Vector DB search results."]
        return list_of_results