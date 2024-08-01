import os
from dataclasses import dataclass
from ._utils import load_json, write_json


@dataclass
class BaseStorage:
    namespace: str
    global_config: dict

    def get_by_id(self, id):
        raise NotImplementedError

    def upsert(self, data: list):
        raise NotImplementedError


@dataclass
class JsonKVStorage(BaseStorage):

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}

    def get_by_id(self, id):
        return self._data[id]

    def upsert(self, data: dict):
        self._data.update(data)
        write_json(self._data, self._file_name)
