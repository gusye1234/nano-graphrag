import asyncio
from contextlib import contextmanager
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Literal
import networkx as nx
from ng_nx import NebulaReader
import numpy as np
from networkx.classes.reportviews import  NodeView
from nano_graphrag._storage.gdb_networkx import NetworkXStorage
from nano_graphrag.prompt import GRAPH_FIELD_SEP
from ng_nx import NebulaWriter
from ng_nx.utils import NebulaGraphConfig

from .._utils import get_workdir_last_folder_name, logger
from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    SingleCommunitySchema,
)

@dataclass
class NebulaGraphStorage(BaseGraphStorage,BaseKVStorage):
    # TODO: consider add @RANK to enable multi-edge per src-->tgt

    # NOTE: client can only be used when space exists.
    client: Any = None # lazy dependency, thus Any typed here.

    # ng_nx
    config: Any = None
    reader: Any = None
    writer_cls: Any = None
    _graph: Any = None
    _graph_homogenous: Any = None

    # DEFAULT SCHEMA
    VID_LENGTH: int = 256
    # We are using a Homogeneous Graph Model for Graph Index
    INIT_EDGE_TYPE: str = "RELATED_TO"
    INIT_EDGE_PROPERTIES: list[dict[str, str]] = field(default_factory=lambda: [
        {"name": "weight", "type": {"type": "float"}, "DEFAULT": 0.0},
        {"name": "description", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "order", "type": {"type": "int"}, "DEFAULT": 1}, 
        {"name": "source_id", "type": {"type": "string"}, "DEFAULT": "''"},
    ])
    
    INIT_EDGE_INDEXES: list[dict[str, str]] = field(default_factory=lambda: [
        {"index_name": "relation_index", "fields_str": "()"},
    ])

    INIT_VERTEX_TYPE: str = "entity"
    INIT_VERTEX_PROPERTIES: list[dict[str, str]] = field(default_factory=lambda: [
        {"name": "entity_name", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "entity_type", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "description", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "source_id", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "clusters", "type": {"type": "string"}, "DEFAULT": "''"},
    ])
    INIT_VERTEX_INDEXES: list[dict[str, str]] = field(default_factory=lambda: [
        {"index_name": "entity_index", "fields_str": "()"},
    ])

    # Schema For Meta Knowledge Graph
    # Community Report Vertex Type and Edge Type
    COMMUNITY_VERTEX_TYPE: str = "community"
    COMMUNITY_VERTEX_PROPERTIES: list[dict[str, str]] = field(default_factory=lambda: [
        {"name": "level", "type": {"type": "int"}, "DEFAULT": -1},
        {"name": "cluster", "type": {"type": "int"}, "DEFAULT": -1},
        {"name": "title", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "report_string", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "report_json", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "chunk_ids", "type": {"type": "string"}, "DEFAULT": "''"},
        {"name": "occurrence", "type": {"type": "float"}, "DEFAULT": 0.0},
        {"name": "sub_communities", "type": {"type": "string"}, "DEFAULT": "''"},
    ])
    COMMUNITY_VERTEX_INDEXES: list[dict[str, str]] = field(default_factory=lambda: [
        {"index_name": "community_vertex_index", "fields_str": "()"},
    ])
    COMMUNITY_EDGE_TYPE: str = "ENTITY_WITHIN_COMMUNITY"
    COMMUNITY_EDGE_PROPERTIES: list[dict[str, str]] = field(default_factory=lambda: [
        {"name": "level", "type": {"type": "int"}, "DEFAULT": -1},
    ])
    COMMUNITY_EDGE_INDEXES: list[dict[str, str]] = field(default_factory=lambda: [
        {"index_name": "community_edge_index", "fields_str": "()"},
    ])

    HEATBEAT_TIME: int = 10
    IMPLEMENTED_PARAM_TYPES: list[type] = field(default_factory=lambda: [str])
    INSERT_BATCH_SIZE: int = 64
    JSON_FIELDS: list[str] = field(default_factory=lambda: ["report_json", "chunk_ids", "sub_communities"])

    @staticmethod
    def _graph_exists(session: Any, space_name: str) -> bool:
        try:
            spaces = session.execute("SHOW SPACES;").column_values("Name")
            return any(space_name == space.as_string() for space in spaces)
        except Exception as e:
            error_message = f"Failed to check if graph space '{space_name}' exists: {e}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e

    @staticmethod
    def _label_exists(session: Any, space_name: str, type: Literal["tag", "edge"], label: str):
        TAGS = "TAGS"
        EDGES = "EDGES"
        try:
            session.execute(f"USE {space_name};")
            rest = session.execute(f"SHOW {TAGS if type == 'tag' else EDGES};").column_values("Name")
            return any(label == l.as_string() for l in rest)
        except Exception as e:
            error_message = f"Failed to check if '{type}' exists in graph space '{space_name}': {e}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e

    @staticmethod
    def _index_exists(session: Any, space_name: str, type: Literal["tag", "edge"], index_name: str):
        try:
            session.execute(f"USE {space_name};")
            rest = session.execute(f"SHOW {type} INDEXES;").column_values("Index Name")
            return any(index_name == i.as_string() for i in rest)
        except Exception as e:
            error_message = f"Failed to check if index '{index_name}' exists in graph space '{space_name}': {e}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e

    async def _create_graph(self, session: Any, space_name: str, delay_time: float = 20, vid_length: int = 256):
        from nebula3.data.ResultSet import ResultSet

        session.execute(
            f"CREATE SPACE IF NOT EXISTS {space_name} (replica_factor = 1, vid_type=FIXED_STRING({vid_length}));"
        )

        backoff_time_counter = 0
        attempt = 0
        while True:
            attempt += 1
            result: ResultSet = session.execute(
                f"DESCRIBE SPACE {space_name}; USE {space_name};"
            )
            if result.is_succeeded():
                logger.info(f"Graph space {space_name} created successfully")
                break
            else:
                if backoff_time_counter < delay_time:
                    backoff_time = 2**attempt
                    backoff_time_counter += backoff_time
                    await asyncio.sleep(backoff_time)
                else:
                    session.release()
                    raise ValueError(
                        f"Graph Space {space_name} creation failed in {backoff_time_counter} seconds"
                    )

    async def _init_nebulagraph_schema(self) -> None:
        with self._session() as session:
            # Ensure graph space exists
            if not self._graph_exists(session, self.space):
                logger.info(f"Creating graph space {self.space}")
                await self._create_graph(session, self.space, vid_length=self.VID_LENGTH)

            # Ensure initial schema exists
            create_flag = False
            if not self._label_exists(session, self.space, "tag", self.INIT_VERTEX_TYPE):
                prop_list = [f"`{prop['name']}` {prop['type']['type']} DEFAULT {prop['DEFAULT']}" for prop in self.INIT_VERTEX_PROPERTIES]
                prop_str = ",".join(prop_list)
                CREATE_TAG_QUERY = f"CREATE TAG IF NOT EXISTS `{self.INIT_VERTEX_TYPE}` ({prop_str})"
                session.execute(CREATE_TAG_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "edge", self.INIT_EDGE_TYPE):
                prop_list = [f"`{prop['name']}` {prop['type']['type']}  DEFAULT {prop['DEFAULT']}" for prop in self.INIT_EDGE_PROPERTIES]
                prop_str = ",".join(prop_list)
                CREATE_EDGE_QUERY = f"CREATE EDGE IF NOT EXISTS `{self.INIT_EDGE_TYPE}` ({prop_str})"
                session.execute(CREATE_EDGE_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "tag", self.COMMUNITY_VERTEX_TYPE):
                prop_list = [f"`{prop['name']}` {prop['type']['type']}  DEFAULT {prop['DEFAULT']}" for prop in self.COMMUNITY_VERTEX_PROPERTIES]
                prop_str = ",".join(prop_list)
                CREATE_TAG_QUERY = f"CREATE TAG IF NOT EXISTS `{self.COMMUNITY_VERTEX_TYPE}` ({prop_str})"
                session.execute(CREATE_TAG_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "edge", self.COMMUNITY_EDGE_TYPE):
                prop_list = [f"`{prop['name']}` {prop['type']['type']}  DEFAULT {prop['DEFAULT']}" for prop in self.COMMUNITY_EDGE_PROPERTIES]
                prop_str = ",".join(prop_list)
                CREATE_EDGE_QUERY = f"CREATE EDGE IF NOT EXISTS `{self.COMMUNITY_EDGE_TYPE}` ({prop_str})"
                session.execute(CREATE_EDGE_QUERY)
                create_flag = True

            if create_flag:
                # Wait for schema changes to propagate
                logger.info(f"Waiting {2 * self.HEATBEAT_TIME + 1}s for schema changes to propagate")
                await asyncio.sleep(2 * self.HEATBEAT_TIME + 1)

                # Verify tag creation
                if not self._label_exists(session, self.space, "tag", self.INIT_VERTEX_TYPE):
                    raise RuntimeError(f"Failed to create tag {self.INIT_VERTEX_TYPE}")

                # Verify edge creation
                if not self._label_exists(session, self.space, "edge", self.INIT_EDGE_TYPE):
                    raise RuntimeError(f"Failed to create edge {self.INIT_EDGE_TYPE}")

                # Ensure Meta Knowledge Graph Schema
                if not self._label_exists(session, self.space, "tag", self.COMMUNITY_VERTEX_TYPE):
                    raise RuntimeError(f"Failed to create tag {self.COMMUNITY_VERTEX_TYPE}")
                if not self._label_exists(session, self.space, "edge", self.COMMUNITY_EDGE_TYPE):
                    raise RuntimeError(f"Failed to create edge {self.COMMUNITY_EDGE_TYPE}")

                logger.info(f"Successfully created initial schema for graph space {self.space}")

            # Ensure initial indexes exists
            create_flag = False
            for tag_index in self.INIT_VERTEX_INDEXES:
                if not self._index_exists(session, self.space, "tag", tag_index['index_name']):
                    session.execute(f"CREATE TAG INDEX IF NOT EXISTS `{tag_index['index_name']}` ON `{self.INIT_VERTEX_TYPE}` {tag_index['fields_str']}")
                    logger.info(f"Created tag index {tag_index['index_name']} on {self.INIT_VERTEX_TYPE}")
                    create_flag = True
            for edge_index in self.INIT_EDGE_INDEXES:
                if not self._index_exists(session, self.space, "edge", edge_index['index_name']):
                    session.execute(f"CREATE EDGE INDEX IF NOT EXISTS `{edge_index['index_name']}` ON `{self.INIT_EDGE_TYPE}` {edge_index['fields_str']}")
                    logger.info(f"Created edge index {edge_index['index_name']} on {self.INIT_EDGE_TYPE}")
                    create_flag = True
            # Ensure Meta Knowledge Graph Indexes
            for tag_index in self.COMMUNITY_VERTEX_INDEXES:
                if not self._index_exists(session, self.space, "tag", tag_index['index_name']):
                    session.execute(f"CREATE TAG INDEX IF NOT EXISTS `{tag_index['index_name']}` ON `{self.COMMUNITY_VERTEX_TYPE}` {tag_index['fields_str']}")
                    logger.info(f"Created tag index {tag_index['index_name']} on {self.COMMUNITY_VERTEX_TYPE}")
                    create_flag = True
            for edge_index in self.COMMUNITY_EDGE_INDEXES:
                if not self._index_exists(session, self.space, "edge", edge_index['index_name']):
                    session.execute(f"CREATE EDGE INDEX IF NOT EXISTS `{edge_index['index_name']}` ON `{self.COMMUNITY_EDGE_TYPE}` {edge_index['fields_str']}")
                    logger.info(f"Created edge index {edge_index['index_name']} on {self.COMMUNITY_EDGE_TYPE}")
                    create_flag = True

            if create_flag:
                # Wait for index creation to complete
                logger.info(f"Waiting {2 * self.HEATBEAT_TIME + 1}s for index creation to complete")
                await asyncio.sleep(2 * self.HEATBEAT_TIME + 1)

                # Verify index creation
                for tag_index in self.INIT_VERTEX_INDEXES:
                    if not self._index_exists(session, self.space, "tag", tag_index['index_name']):
                        raise RuntimeError(f"Failed to create tag index {tag_index['index_name']}")
                for edge_index in self.INIT_EDGE_INDEXES:
                    if not self._index_exists(session, self.space, "edge", edge_index['index_name']):
                        raise RuntimeError(f"Failed to create edge index {edge_index['index_name']}")

                # Ensure Meta Knowledge Graph Indexes
                for tag_index in self.COMMUNITY_VERTEX_INDEXES:
                    if not self._index_exists(session, self.space, "tag", tag_index['index_name']):
                        raise RuntimeError(f"Failed to create tag index {tag_index['index_name']}")
                for edge_index in self.COMMUNITY_EDGE_INDEXES:
                    if not self._index_exists(session, self.space, "edge", edge_index['index_name']):
                        raise RuntimeError(f"Failed to create edge index {edge_index['index_name']}")

    def get_graphd_addresses(self) -> list[tuple[str, int]]:
        graphd_host_address: list[tuple[str, int]] = []
        for host in self.graphd_hosts.split(","):
            # sanity check
            if ":" not in host:
                raise ValueError(f"Invalid graphd host {host}, should be host:port")
            host, port = host.split(":")
            if not port.isdigit():
                raise ValueError(f"Invalid port {port} in host {host}")
            if int(port) < 0 or int(port) > 65535:
                raise ValueError(f"Invalid port {port} in host {host}")
            if not host:
                raise ValueError(f"Invalid host {host}")
            graphd_host_address.append((host, int(port)))
        return graphd_host_address

    @contextmanager
    def _session(self) -> Any:
        """
        Only used for space creation and schema initialization.
        """
        from nebula3.Config import Config, SSL_config
        from nebula3.gclient.net.ConnectionPool import ConnectionPool

        conn_pool = ConnectionPool()
        graphd_host_address_list = self.get_graphd_addresses()
        try:

            conn_pool.init(
                graphd_host_address_list,
                configs=Config(),
                ssl_conf=SSL_config() if self.use_tls else None
            )

            session = conn_pool.get_session(self.username, self.password)
            try:
                yield session
            finally:
                session.release()
        finally:
            conn_pool.close()

    def _initialize_session_pool(self) -> None:
        """
        Initialize and set up the session pool as a singleton.
        The space is created and schema initialized before this method is called.
        """
        from nebula3.gclient.net.SessionPool import SessionPool
        from nebula3.Config import SessionPoolConfig, SSL_config

        graphd_host_address = self.get_graphd_addresses()
        try:
            session_pool = SessionPool(
                self.username,
                self.password,
                self.space,
                graphd_host_address
            )

            session_pool_config = SessionPoolConfig()
            session_pool.init(
                session_pool_config,
                ssl_configs=SSL_config() if self.use_tls else None
            )
            self.client: SessionPool = session_pool
        except Exception as e:
            raise RuntimeError(f"Failed to initialize session pool: {e}") from e

    def __post_init__(self):
        self.space: str = get_workdir_last_folder_name(self.global_config["working_dir"])
        self.use_tls: bool = self.global_config["addon_params"].get("use_tls", False)
        self.graphd_hosts: str = self.global_config["addon_params"].get("graphd_hosts", None)
        self.metad_hosts: str = self.global_config["addon_params"].get("metad_hosts", None)
        self.username: str = self.global_config["addon_params"].get("username", "root")
        self.password: str = self.global_config["addon_params"].get("password", "nebula")
        self.VID_LENGTH: int = 256

        if not self.graphd_hosts or not self.metad_hosts:
            raise ValueError("Missing required connection information: graphd_hosts and metad_hosts not provided")

        asyncio.run(self._init_nebulagraph_schema())

        self.config = NebulaGraphConfig(
            space=self.space,
            graphd_hosts=self.graphd_hosts,
            metad_hosts=self.metad_hosts
        )
        self.reader = MyNebulaReader(
            edges=[self.INIT_EDGE_TYPE],
            properties=[[prop['name'] for prop in self.INIT_EDGE_PROPERTIES]],
            nebula_config=self.config,
            limit=1000000
        )
        self.writer_cls = NebulaWriter
        self._graph = None
        self._initialize_session_pool()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }


    async def has_node(self, node_id: str) -> bool:
        if node_id is None or not isinstance(node_id, str):
            raise ValueError(f"Invalid node_id {node_id}")
        if node_id == "":
            raise ValueError(f"Invalid node_id {node_id}")
        
        if len(node_id.encode('utf-8')) > self.VID_LENGTH:
            return False

        try:
            result_n = self.client.execute_py(
                f"MATCH (n:{self.INIT_VERTEX_TYPE}) WHERE id(n) == $node_id RETURN n;", params={"node_id": node_id}
            ).column_values("n")
            return len(result_n) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if node {node_id} exists: {e}") from e

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        if target_node_id is None or not isinstance(target_node_id, str) or target_node_id == "":
            raise ValueError(f"Invalid target_node_id {target_node_id}")
        if len(source_node_id.encode('utf-8')) > self.VID_LENGTH or len(target_node_id.encode('utf-8')) > self.VID_LENGTH:
            return False

        (sorted_source_node_id, sorted_target_node_id) = sorted([source_node_id, target_node_id])

        try:
            result = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]-(m) WHERE id(n) == $source_node_id AND id(m) == $target_node_id RETURN e",
                params={"source_node_id": sorted_source_node_id, "target_node_id": sorted_target_node_id}
            ).column_values("e")
            return len(result) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if edge exists between {sorted_source_node_id} and {sorted_target_node_id}: {e}") from e


    async def get_node(self, node_id: str) -> Union[dict, NodeView, None]:
        if node_id is None or not isinstance(node_id, str) or node_id == "":
            raise ValueError(f"Invalid node_id {node_id}")
        if len(node_id.encode('utf-8')) > self.VID_LENGTH:
            return None
        
        try:
            result = self.client.execute_py(
                f"MATCH (n:{self.INIT_VERTEX_TYPE}) WHERE id(n) == $node_id RETURN n",
                params={"node_id": node_id}
            ).column_values("n")

            if not result:
                return None
            
            node = result[0].as_node()
            return {
                "id": node_id,
                **{k: v.cast() for k, v in node.properties().items() if not (k == 'clusters' and v.cast() == '')} # compatibility for _find_most_related_community_from_entities
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get node {node_id}: {e}") from e

    async def node_degree(self, node_id: str) -> int:
        if node_id is None or not isinstance(node_id, str) or node_id == "":
            raise ValueError(f"Invalid node_id {node_id}")
        try:
            result = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]-() WHERE id(n) == $node_id RETURN count(e) AS Degree",
                params={"node_id": node_id}
            ).column_values("Degree")
            if not result:
                return 0
            return result[0].cast()
        except Exception as e:
            raise RuntimeError(f"Failed to get node degree for {node_id}: {e}") from e

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        ref:https://github.com/microsoft/graphrag/blob/718d1ef441137a6aed3bb9c445aabbdf612c03b9/graphrag/index/verbs/graph/compute_edge_combined_degree.py
        it is defined as the number of neighbors of the source node plus the number of neighbors of the target node.
        """
        if src_id is None or not isinstance(src_id, str) or src_id == "":
            raise ValueError(f"Invalid src_id {src_id}")
        if tgt_id is None or not isinstance(tgt_id, str) or tgt_id == "":
            raise ValueError(f"Invalid tgt_id {tgt_id}")

        try:
            source_degree = await self.node_degree(src_id)
            target_degree = await self.node_degree(tgt_id)
            return source_degree + target_degree
        except Exception as e:
            raise RuntimeError(f"Failed to compute edge degree between {src_id} and {tgt_id}: {e}") from e
        

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        if target_node_id is None or not isinstance(target_node_id, str) or target_node_id == "":
            raise ValueError(f"Invalid target_node_id {target_node_id}")
        
        # NebulaGraph is unordered, so we need to sort the source and target node ids
        (sorted_source_node_id, sorted_target_node_id) = sorted([source_node_id, target_node_id])

        try:
            result: list[dict] = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]-(m) WHERE id(n) == $src_id AND id(m) == $tgt_id RETURN e",
                params={"src_id": sorted_source_node_id, "tgt_id": sorted_target_node_id}
            ).as_primitive()

            if not result:
                return None

            if len(result) > 1:
                logger.warning(f"Found multiple edges between {source_node_id} and {target_node_id}")
                
            edge_primitive = result[0]
            if not edge_primitive or not edge_primitive.get('e'):
                return None
            edge_props = {k:v.cast()  for k,v in edge_primitive['e']['props'].items()}
            edge = {
                'source_node_id': source_node_id,
                'target_node_id': target_node_id,
                **edge_props
            }
            return edge

        except Exception as e:
            raise RuntimeError(f"Failed to get edge between {source_node_id} and {target_node_id}: {e}") from e

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        try:
            result = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]-(m) WHERE id(n) == $src_id RETURN id(n) AS source, id(m) AS target",
                params={"src_id": source_node_id}
            ).as_primitive()

            if not result:
                return []
            
            edges = [(edge['source'], edge['target']) for edge in result]
            return edges if edges else None
        except Exception as e:
            raise RuntimeError(f"Failed to get edges for node {source_node_id}: {e}") from e

    async def upsert_node(self, node_id: str, node_data: dict[str, str], label: Optional[str] = None):
        if node_id is None or not isinstance(node_id, str) or node_id == "":
            raise ValueError(f"Invalid node_id {node_id}")
        if node_data is None or not isinstance(node_data, dict):
            raise ValueError(f"Invalid node_data {node_data}")
        if not node_data:
            raise ValueError(f"Invalid node_data {node_data}")
        if len(node_id.encode('utf-8')) > self.VID_LENGTH:
            return

        from uuid import uuid4
        label = label or self.INIT_VERTEX_TYPE
        if 'entity_name' not in node_data:
            node_data['entity_name'] = node_id
        prop_all_names = list(node_data.keys())
        prop_name = ",".join(
            [prop for prop in prop_all_names if node_data[prop] is not None]
        )
        props_ngql: list[str] = []
        props_map: dict[str, Any] = {}
        for prop in prop_all_names:
            if node_data[prop] is None:
                continue
            if any([isinstance(node_data[prop], t) for t in self.IMPLEMENTED_PARAM_TYPES]):
                new_key = "k_" + uuid4().hex
                props_ngql.append(f"${new_key}")
                props_map[new_key] = node_data[prop]
            else:
                props_ngql.append(str(node_data[prop]))
        prop_val = ",".join(props_ngql)

        query = (
            f"INSERT VERTEX `{label}`({prop_name}) "
            f"  VALUES '{escape_bucket(node_id)}':({prop_val});\n"
        )
        logger.debug(f"upsert_node()\nDML query: {query}")
        result = self.client.execute_py(query, props_map)

        if not result.is_succeeded():
            raise RuntimeError(f"Failed to upsert node {escape_bucket(node_id)}: {result} with query {query}")

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str], label: Optional[str] = None):
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        if target_node_id is None or not isinstance(target_node_id, str) or target_node_id == "":
            raise ValueError(f"Invalid target_node_id {target_node_id}")
        if edge_data is None or not isinstance(edge_data, dict):
            raise ValueError(f"Invalid edge_data {edge_data}")
        if not edge_data:
            raise ValueError(f"Invalid edge_data {edge_data}")

        if len(source_node_id.encode('utf-8')) > self.VID_LENGTH or len(target_node_id.encode('utf-8')) > self.VID_LENGTH:
            return

        (sorted_source_node_id, sorted_target_node_id) = sorted([source_node_id, target_node_id])

        from uuid import uuid4
        label = label or self.INIT_EDGE_TYPE

        prop_all_names = list(edge_data.keys())
        prop_name = ",".join(
            [f"`{prop}`" for prop in prop_all_names if edge_data[prop] is not None]
        )
        props_ngql: list[str] = []
        props_map: dict[str, Any] = {}
        for prop in prop_all_names:
            if edge_data[prop] is None:
                continue
            if any([isinstance(edge_data[prop], t) for t in self.IMPLEMENTED_PARAM_TYPES]):
                new_key = "k_" + uuid4().hex
                props_ngql.append(f"${new_key}")
                props_map[new_key] = edge_data[prop]
            else:
                props_ngql.append(str(edge_data[prop]))
        prop_val = ",".join(props_ngql)
        query = (
            f"INSERT EDGE `{label}`({prop_name}) "
            f"  VALUES '{escape_bucket(sorted_source_node_id)}'->'{escape_bucket(sorted_target_node_id)}':({prop_val});\n"
        )
        logger.debug(f"upsert_edge()\nDML query: {query}")
        result = self.client.execute_py(query, props_map)
        if not result.is_succeeded():
            raise RuntimeError(f"Failed to upsert edge between {escape_bucket(sorted_source_node_id)} and {escape_bucket(sorted_target_node_id)}: {result} with query {query}")


    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()


    async def _cluster_data_to_graph(self, cluster_data: dict[str, list[dict[str, str]]]):
        # community node: (level, cluster), with id cluster_{cluster}
        # cluster_{cluster} is the key, and value is a list of (level, cluster)
        data: dict[str, list[int, int]] = defaultdict(list)
        # entity --> cluster
        # (entity_id, cluster_id, level)
        edges: list[tuple[int, int, int]] = []
        rank = 0 # just a placeholder, no used yet
        for node_id, clusters in cluster_data.items():
            for cluster in clusters:
                cluster_id = cluster["cluster"]
                level = cluster["level"]
                cluster_node_id = f'{cluster_id}'
                if cluster_node_id not in data:
                    data[cluster_node_id] = [level, cluster_id]
                edges.append((escape_bucket(node_id), cluster_node_id, rank, [level]))
        
        # add clusters info to entity node , for compatibility 
        async def update_node_clusters(node_id, clusters):
            if await self.has_node(node_id):
                clusters_json = json.dumps(clusters, ensure_ascii=False)
                update_query = (
                    f"UPDATE VERTEX ON {self.INIT_VERTEX_TYPE} '{escape_bucket(node_id)}' "
                    f"SET clusters = '{clusters_json}';"
                )
                result = self.client.execute_py(update_query) 
                if not result.is_succeeded():
                    raise RuntimeError(f"update {node_id} clusters failed: {result.error_msg()}")

        update_tasks = []
        for node_id, clusters in cluster_data.items():
            update_tasks.append(update_node_clusters(node_id, clusters))

        await asyncio.gather(*update_tasks)

        # Community Vertex data
        ng_nx_community_writer = self.writer_cls(
            data=data,
            nebula_config=self.config,
        )
        ng_nx_community_writer.set_options(
            label=self.COMMUNITY_VERTEX_TYPE,
            properties=["level", "cluster"],
            write_mode="insert",
            sink="nebulagraph_vertex",
        )
        ng_nx_community_writer.write()

        ng_nx_community_edge_writer = self.writer_cls(
            data=edges,
            nebula_config=self.config,
        )
        ng_nx_community_edge_writer.set_options(
            label=self.COMMUNITY_EDGE_TYPE,
            properties=["level"],
            write_mode="insert",
            sink="nebulagraph_edge",
        )
        ng_nx_community_edge_writer.write()

        return


    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        # TODO: introduce Cache mechanism for this.
        self._graph = self.reader.read()
        self._graph_homogenous = nx.Graph(self._graph)

        graph = NetworkXStorage.stable_largest_connected_component(self._graph_homogenous)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=self.global_config["max_graph_cluster_size"],
            random_seed=self.global_config["graph_cluster_seed"],
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)

        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        await self._cluster_data_to_graph(node_communities)

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)


        communities_result = self.client.execute_py(
            f"MATCH (n:{self.COMMUNITY_VERTEX_TYPE}) RETURN n"
        ).column_values("n")

        for community in communities_result:
            community_data = community.as_node()
            community_id = community_data.get_id().cast()
            community_properties = {k: v.cast() for k, v in community_data.properties().items()}
            level = community_properties.get("level")
            community_key = str(community_properties.get("cluster"))
            title = community_id

            node_query = f"MATCH (n:{self.INIT_VERTEX_TYPE})-[:{self.COMMUNITY_EDGE_TYPE}]->(c:{self.COMMUNITY_VERTEX_TYPE}) WHERE id(c) == '{community_id}' RETURN n;"
            nodes_in_community = self.client.execute_py(node_query)
            nodes = set()
            edges = set()
            chunk_ids = set()
            for node in nodes_in_community.column_values("n"):
                node_data = node.as_node()
                node_id = node_data.get_id().cast()
                nodes.add(node_id)
                node_properties = {k: v.cast() for k, v in node_data.properties().items()}
                chunk_ids.update(node_properties.get("source_id", "").split(GRAPH_FIELD_SEP))
            
                edge_query = f"MATCH (n:{self.INIT_VERTEX_TYPE})-[:{self.INIT_EDGE_TYPE}]-(m:{self.INIT_VERTEX_TYPE}) WHERE id(n) == '{escape_bucket(node_id)}' RETURN m;"
                edges_in_node = self.client.execute_py(edge_query)
                for edge in edges_in_node.column_values("m"):
                    dst_node = edge.as_node()
                    dst_node_id = dst_node.get_id().cast()
                    edges.add((node_id, dst_node_id))

            results[community_key].update(
                level=level,
                title=title,
                chunk_ids=chunk_ids,
                nodes=nodes,
                edges=edges,
            )

            levels[level].add(community_key)
            max_num_ids = max(max_num_ids, len(chunk_ids))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]

            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]
        
        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)
    
    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        # TODO: persist node2vec/graph embedding to NebulaGraph
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        # TOOD: introduce Cache mechanism for this to scale even better.
        self._graph = self.reader.read()
        self._graph_homogenous = nx.Graph(self._graph)

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    async def index_done_callback(self):
        # close the client
        self.client.close()
        # TODO: introduce cache mechnism, then we could leverage this callback

    ## ↓ KV Storage Implementation ↓ ##

    async def all_keys(self) -> list[str]:
        communities_result = self.client.execute_py(
            f"MATCH (n:{self.COMMUNITY_VERTEX_TYPE}) RETURN n"
        ).column_values("n")
        return [c.as_node().get_id().cast() for c in communities_result]

    async def get_by_id(self, id):
        try:
            result = self.client.execute_py(
                f"MATCH (n:{self.COMMUNITY_VERTEX_TYPE}) WHERE id(n) == $id RETURN n",
                params={"id": id}
            ).column_values("n")

            if not result:
                return None
            
            node = result[0].as_node()
            properties = {k: v.cast() for k, v in node.properties().items()}
            properties = self._parse_json_fields(properties, self.JSON_FIELDS)
            return properties
        except Exception as e:
            raise RuntimeError(f"Failed to get community {id}: {e}") from e
        
    async def get_by_ids(self, ids, fields=None):
        try:
            id_list = ", ".join([f"'{id}'" for id in ids])
            query = f"MATCH (n:{self.COMMUNITY_VERTEX_TYPE}) WHERE id(n) IN [{id_list}] RETURN n"
            result = self.client.execute_py(query).column_values("n")
            
            communities = []
            for node in result:
                properties = {k: v.cast() for k, v in node.as_node().properties().items()}
                properties = self._parse_json_fields(properties, self.JSON_FIELDS)

                if fields:
                    properties = {k: v for k, v in properties.items() if k in fields}
                communities.append(properties)
            
            ordered_communities = []
            for id in ids:
                community = next((c for c in communities if c.get('id') == id), None)
                ordered_communities.append(community)
            
            return ordered_communities
        except Exception as e:
            raise RuntimeError(f"Failed to get communities: {e}") from e
        
    async def filter_keys(self, data: list[str]) -> set[str]:
        raise NotImplementedError
    
    async def upsert(self, data: dict[str, dict]):
        async def upsert_one(id , properties):
            try:
                update_properties = {
                    'report_json': properties.get('report_json', '{}'),
                    'chunk_ids': properties.get('chunk_ids', '[]'),
                    'sub_communities': properties.get('sub_communities', '[]'),
                    'occurrence': properties.get('occurrence', 0.0),
                    'title': properties.get('report_json', '{}').get('title', ''),
                    'report_string': properties.get('report_string', '')
                }

                update_properties = self._dump_json_fields(update_properties, self.JSON_FIELDS)

                update_query = f"UPDATE VERTEX ON {self.COMMUNITY_VERTEX_TYPE} '{id}' SET report_json = $report_json, chunk_ids = $chunk_ids, sub_communities = $sub_communities, occurrence = $occurrence, title = $title, report_string = $report_string"

                result = self.client.execute_py(update_query, params=update_properties)
                if not result.is_succeeded():
                    raise RuntimeError(f"Update community {id} Failed: {result.error_msg()}")

            except Exception as e:
                raise RuntimeError(f"Update community {id} Failed: {e}") from e

        tasks = []
        for id, properties in data.items():
            task = upsert_one(id, properties)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        logger.info(f"Successfully updated {len(data)} communities")

    async def drop(self):
        try:
            # Delete all communitites vertices and edges
            delete_query = f"LOOKUP ON {self.COMMUNITY_VERTEX_TYPE} YIELD id(vertex) AS ID | DELETE VERTEX  $-.ID WITH EDGE;"
            result = self.client.execute_py(delete_query)
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to delete community vertices: {result.error_msg()}")
            
            logger.info("Successfully dropped all community vertices and edges")
        except Exception as e:
            raise RuntimeError(f"Failed to drop community data: {e}") from e
        
    def _parse_json_fields(self, properties, json_fields):
        """
        Parse the json fields in the properties.
        """
        for k, v in properties.items():
            if k in json_fields:
                properties[k] = json.loads(v)
        return properties
    
    def _dump_json_fields(self, properties, json_fields):
        """
        Dump the json fields in the properties.
        """
        for k, v in properties.items():
            if k in json_fields:
                properties[k] = json.dumps(v,ensure_ascii=False)
        return properties




class MyNebulaReader(NebulaReader):
    """
    If the property has the name 'order', it will cause an error. 
    Therefore, it needs to be enclosed with backticks.
    """
    def read(self):
        from ng_nx.utils import result_to_df
        with self.connection_pool.session_context(
            self.nebula_user, self.nebula_password
        ) as session:
            assert session.execute(
                f"USE {self.space}"
            ).is_succeeded(), f"Failed to use space {self.space}"
            result_list = []
            g = nx.MultiDiGraph()
            for i in range(len(self.edges)):
                edge = self.edges[i]
                properties = self.properties[i]
                properties_query_field = ""
                for property in properties:
                    properties_query_field += f", e.`{property}` AS `{property}`"
                if self.with_rank:
                    properties_query_field += ", rank(e) AS `__rank__`"
                result = session.execute(
                    f"MATCH ()-[e:`{edge}`]->() RETURN src(e) AS src, dst(e) AS dst{properties_query_field} LIMIT {self.limit}"
                )
                # print(f'query: MATCH ()-[e:`{edge}`]->() RETURN src(e) AS src, dst(e) AS dst{properties_query_field} LIMIT {self.limit}')
                # print(f"Result: {result}")
                assert result.is_succeeded()
                result_list.append(result)

            # merge all result
            for i, result in enumerate(result_list):
                _df = result_to_df(result)
                # TBD, consider add label of edge
                properties = self.properties[i] if self.properties[i] else None
                if self.with_rank:
                    properties = properties + ["__rank__"]
                    _g = nx.from_pandas_edgelist(
                        _df,
                        "src",
                        "dst",
                        properties,
                        create_using=nx.MultiDiGraph(),
                        edge_key="__rank__",
                    )
                else:
                    _g = nx.from_pandas_edgelist(
                        _df,
                        "src",
                        "dst",
                        properties,
                        create_using=nx.MultiDiGraph(),
                    )
                g = nx.compose(g, _g)
            return g

def escape_bucket(input):
    if isinstance(input,str):
        return input.replace("'", "\\'").replace('"', '\\"')
    elif isinstance(input,list):
        return [escape_bucket(i) for i in input]
    elif isinstance(input,dict):
        return {k: escape_bucket(v) for k, v in input.items()}
    else:
        return input