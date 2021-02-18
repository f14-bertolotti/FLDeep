import json
import pathlib

class Configuration:
    def __init__(self, configuration_path):
        self.path = pathlib.Path(configuration_path)
        self.load()
    def load(self):
        self.configuration = json.loads(self.path.read_text())
    def __getitem__(self, key):
        return self.configuration[key]["value"]
    def __getattr__(self, key):
        return self.configuration[key]["value"]
