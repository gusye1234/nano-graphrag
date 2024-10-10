import json
from dataclasses import dataclass, field

from ..base import BaseKVStorage
import redis
from redis.exceptions import ConnectionError
from .._utils import get_workdir_last_folder_name, logger

@dataclass
class RedisKVStorage(BaseKVStorage):
    _redis: redis.Redis = field(init=False, repr=False, compare=False)
    def __post_init__(self):
        try:
            host = self.global_config["addon_params"].get("redis_host", "localhost")
            port = self.global_config["addon_params"].get("redis_port", "6379")
            user = self.global_config["addon_params"].get("redis_user", None)
            password = self.global_config["addon_params"].get("redis_password", None)
            db = self.global_config["addon_params"].get("redis_db", 0)
            self._redis = redis.Redis(host=host, port=port, username=user, password=password, db=db)
            self._redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except ConnectionError:
            logger.error(f"Failed to connect to Redis at {host}:{port}")
            raise

        self._namespace = f"kv_store_{get_workdir_last_folder_name(self.global_config['working_dir'])}"
        logger.info(f"Initialized Redis KV storage for namespace: {self._namespace}")

    async def all_keys(self) -> list[str]:
        return [key.decode().split(':', 1)[1] for key in self._redis.keys(f"{self._namespace}:*")]

    async def index_done_callback(self):
        # Redis automatically persists data, so no explicit action needed
        pass

    async def get_by_id(self, id):
        value = self._redis.get(f"{self._namespace}:{id}")
        return json.loads(value) if value else None

    async def get_by_ids(self, ids, fields=None):
        pipeline = self._redis.pipeline()
        for id in ids:
            pipeline.get(f"{self._namespace}:{id}")
        values = pipeline.execute()
        
        results = []
        for value in values:
            if value:
                data = json.loads(value)
                if fields:
                    results.append({k: v for k, v in data.items() if k in fields})
                else:
                    results.append(data)
            else:
                results.append(None)
        return results

    async def filter_keys(self, data: list[str]) -> set[str]:
        pipeline = self._redis.pipeline()
        for key in data:
            pipeline.exists(f"{self._namespace}:{key}")
        exists = pipeline.execute()
        return set([key for key, exists in zip(data, exists) if not exists])

    async def upsert(self, data: dict[str, dict]):
        pipeline = self._redis.pipeline()
        for key, value in data.items():
            pipeline.set(f"{self._namespace}:{key}", json.dumps(value,ensure_ascii=False))
        pipeline.execute()

    async def drop(self):
        keys = self._redis.keys(f"{self._namespace}:*")
        if keys:
            self._redis.delete(*keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_redis']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()