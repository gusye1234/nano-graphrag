import os
from dataclasses import dataclass
from ._utils import load_json, write_json


@dataclass
class BaseStorage:
    namespace: str
    global_config: dict

    def get_by_id(self, id):
        raise NotImplementedError

    def update(self, data: list):
        raise NotImplementedError


@dataclass
class JsonKVStorage(BaseStorage):

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"{self.namespace}.json")
        self._data = load_json(self._file_name) or {}

    def get_by_id(self, id):
        return self._data[id]

    def update(self, data: dict):
        self._data.update(data)
        write_json(self._data, self._file_name)
