import asyncio
import html
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Union, cast, Literal, Callable, Dict, Optional

import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView

import numpy as np
from nano_vectordb import NanoVectorDB

from ._utils import load_json, logger, write_json
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
)
from .prompt import GRAPH_FIELD_SEP


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}



@dataclass
class NebulaGraphIndexStorage(BaseKVStorage):
    """
    NebulaGraphIndexStorage is a storage that uses NebulaGraph as the underlying storage.

    GraphRAG Indexed data is natively stored in NebulaGraph, so we map different "namespaces" under the KV abstraction
    to different Graph Vertex TAGs and Edge TYPES(when applicable).

    For full_docs, we have TAG "__Document"
    For text_chunks, we have TAG "__Chunk" and EDGE type "DOC_WITH_CHUNK"
    For community_reports, we have TAG "__Community" and EDGE type "ENTITY_WITHIN_COMMUNITY"
    """
    def __post_init__(self):
        self.nebula_storage = NebulaGraphStorage(
            namespace=self.namespace,
            global_config=self.global_config
        )

    async def all_keys(self) -> list[str]:
        # Return all keys that are in the NebulaGraph on given namespace
        raise NotImplementedError("all_keys() is not implemented for NebulaGraphIndexStorage")

    async def index_done_callback(self):
        await self.nebula_storage.index_done_callback()

    async def get_by_id(self, id):
        # Return dict for given id, we just need to fetch data from NebulaGraph
        # and cast it to dict
        raise NotImplementedError("get_by_id() is not implemented for NebulaGraphIndexStorage")

    async def get_by_ids(self, ids, fields=None):
        # Return list of dict for given ids, we just need to fetch data from NebulaGraph
        # and cast it to dict
        # if fields is not None, we need to return dict with only the fields as whitelisted
        raise NotImplementedError("get_by_ids() is not implemented for NebulaGraphIndexStorage")

    async def filter_keys(self, data: list[str]) -> set[str]:
        # Just return the keys that are not in the NebulaGraph
        raise NotImplementedError("filter_keys() is not implemented for NebulaGraphIndexStorage")

    async def upsert(self, data: dict[str, dict]):
        if self.namespace == "full_docs":
            # Implement full_docs specific logic
            for doc_id, doc_data in data.items():
                await self.nebula_storage.upsert_node(doc_id, {"content": doc_data["content"], "type": "full_doc"})
        elif self.namespace == "text_chunks":
            # Implement text_chunks specific logic
            for chunk_id, chunk_data in data.items():
                await self.nebula_storage.upsert_node(chunk_id, {"content": chunk_data["content"], "type": "text_chunk"})
                await self.nebula_storage.upsert_edge(chunk_data["full_doc_id"], chunk_id, {"type": "DOC_WITH_CHUNK"})
        elif self.namespace == "community_reports":
            # Implement community_reports specific logic
            for report_id, report_data in data.items():
                await self.nebula_storage.upsert_node(report_id, {"content": json.dumps(report_data), "type": "community_report"})
        else:
            raise ValueError(f"Unsupported namespace for NebulaGraphIndexStorage: {self.namespace}")

    async def drop(self):
        # Implement based on NebulaGraph query
        raise NotImplementedError("drop() is not implemented for NebulaGraphIndexStorage")


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):

    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding, top_k=top_k, better_than_threshold=0.2
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

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
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
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

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
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
        self._cluster_data_to_subgraphs(node_communities)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

@dataclass
class NebulaGraphStorage(BaseGraphStorage):
    # TODO, implement configration via global_config["addon_params"]
    # credentials
    space: str = "nano_graphrag"
    use_tls: bool = False
    graphd_hosts: str = "127.0.0.1:9669"
    metad_hosts: str = "127.0.0.1:9559"
    username: str = os.environ.get("NEBULA_USER", "root")
    password: str = os.environ.get("NEBULA_PASSWORD", "nebula")

    # nebula3-python
    # NOTE: client can only be used when space exists.
    client: Any = None # lazy dependency, thus Any typed here.

    # ng_nx
    config: Any = None
    reader: Any = None
    writer_cls: Any = None
    _graph: Any = None
    _graph_homogenous: Any = None

    # DEFAULT SCHEMA
    # We are using a Homogeneous Graph Model for Graph Index
    INIT_EDGE_TYPE: str = "RELATED_TO"
    INIT_EDGE_PROPERTIES: list[dict[str, str]] = [
        {"name": "weight", "type": {"type": "float"}},
        {"name": "description", "type": {"type": "string"}},
        {"name": "rank", "type": {"type": "int"}}, # TODO: revisit if leveraging NebulaGraph Native edge.rank
    ]
    INIT_EDGE_INDEXES: list[dict[str, str]] = [
        {"index_name": "relation_rank_index", "fields_str": "(rank)"},
        {"index_name": "relation_weight_index", "fields_str": "(weight)"},
    ]

    INIT_VERTEX_TYPE: str = "entity"
    INIT_VERTEX_PROPERTIES: list[dict[str, str]] = [
        {"name": "name", "type": {"type": "string"}},
        {"name": "entity_type", "type": {"type": "string"}},
        {"name": "description", "type": {"type": "string"}},
        {"name": "source_id", "type": {"type": "string"}}, # TODO: this is duplicated but current implementation requires it.
    ]
    INIT_VERTEX_INDEXES: list[dict[str, str]] = [
        {"index_name": "entity_name_index", "fields_str": "(name(256))"},
        {"index_name": "entity_entity_type_index", "fields_str": "(entity_type(256))"},
    ]

    # Schema For Meta Knowledge Graph

    # Community Report Vertex Type and Edge Type
    COMMUNITY_VERTEX_TYPE: str = "community"
    COMMUNITY_VERTEX_PROPERTIES: list[dict[str, str]] = [
        {"name": "level", "type": {"type": "int"}},
        {"name": "cluster", "type": {"type": "int"}},
        {"name": "title", "type": {"type": "string"}},
        {"name": "summary", "type": {"type": "string"}},
    ]
    COMMUNITY_VERTEX_INDEXES: list[dict[str, str]] = [
        {"index_name": "community_level_index", "fields_str": "(level)"},
        {"index_name": "community_cluster_index", "fields_str": "(cluster)"},
    ]
    COMMUNITY_EDGE_TYPE: str = "ENTITY_WITHIN_COMMUNITY"
    COMMUNITY_EDGE_PROPERTIES: list[dict[str, str]] = [
        {"name": "level", "type": {"type": "int"}},
    ]

    # Findings
    FINDING_VERTEX_TYPE: str = "finding"
    FINDING_VERTEX_PROPERTIES: list[dict[str, str]] = [
        {"name": "summary", "type": {"type": "string"}},
        {"name": "explanation", "type": {"type": "string"}},
    ]
    COMMUNITY_FINDING_EDGE_TYPE: str = "COMMUNITY_WITH_FINDING"
    COMMUNITY_FINDING_EDGE_PROPERTIES: list[dict[str, str]] = []

    # DEFAULTS
    HEATBEAT_TIME: int = 10
    IMPLEMENTED_PARAM_TYPES: list[type] = [str]
    INSERT_BATCH_SIZE: int = 64

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
            return any(label == label.as_string() for label in rest)
        except Exception as e:
            error_message = f"Failed to check if '{type}' exists in graph space '{space_name}': {e}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e

    @staticmethod
    def _index_exists(session: Any, space_name: str, type: Literal["tag", "edge"], index_name: str):
        try:
            session.execute(f"USE {space_name};")
            rest = session.execute(f"SHOW {type} INDEXES;").column_values("Index Name")
            return any(index_name == index_name.as_string() for index_name in rest)
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

        session.release()


    async def _init_nebulagraph_schema(self) -> None:
        with self._session() as session:
            # Ensure graph space exists
            if not self._graph_exists(session, self.space):
                logger.info(f"Creating graph space {self.space}")
                self._create_graph(session, self.space)

            # Ensure initial schema exists
            create_flag = False
            if not self._label_exists(session, self.space, "tag", self.INIT_VERTEX_TYPE):
                CREATE_TAG_QUERY = f"CREATE TAG IF NOT EXISTS `{self.INIT_VERTEX_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.INIT_VERTEX_PROPERTIES])})"
                session.execute(CREATE_TAG_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "edge", self.INIT_EDGE_TYPE):
                CREATE_EDGE_QUERY = f"CREATE EDGE IF NOT EXISTS `{self.INIT_EDGE_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.INIT_EDGE_PROPERTIES])})"
                session.execute(CREATE_EDGE_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "tag", self.COMMUNITY_VERTEX_TYPE):
                CREATE_TAG_QUERY = f"CREATE TAG IF NOT EXISTS `{self.COMMUNITY_VERTEX_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.COMMUNITY_VERTEX_PROPERTIES])})"
                session.execute(CREATE_TAG_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "edge", self.COMMUNITY_EDGE_TYPE):
                CREATE_EDGE_QUERY = f"CREATE EDGE IF NOT EXISTS `{self.COMMUNITY_EDGE_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.COMMUNITY_EDGE_PROPERTIES])})"
                session.execute(CREATE_EDGE_QUERY)
                create_flag = True
            # Ensure Meta Knowledge Graph Schema
            if not self._label_exists(session, self.space, "tag", self.COMMUNITY_VERTEX_TYPE):
                CREATE_TAG_QUERY = f"CREATE TAG IF NOT EXISTS `{self.COMMUNITY_VERTEX_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.COMMUNITY_VERTEX_PROPERTIES])})"
                session.execute(CREATE_TAG_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "edge", self.COMMUNITY_EDGE_TYPE):
                CREATE_EDGE_QUERY = f"CREATE EDGE IF NOT EXISTS `{self.COMMUNITY_EDGE_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.COMMUNITY_EDGE_PROPERTIES])})"
                session.execute(CREATE_EDGE_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "tag", self.FINDING_VERTEX_TYPE):
                CREATE_TAG_QUERY = f"CREATE TAG IF NOT EXISTS `{self.FINDING_VERTEX_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.FINDING_VERTEX_PROPERTIES])})"
                session.execute(CREATE_TAG_QUERY)
                create_flag = True
            if not self._label_exists(session, self.space, "edge", self.COMMUNITY_FINDING_EDGE_TYPE):
                CREATE_EDGE_QUERY = f"CREATE EDGE IF NOT EXISTS `{self.COMMUNITY_FINDING_EDGE_TYPE}` ({', '.join([f'`{prop['name']}` {prop['type']}' for prop in self.COMMUNITY_FINDING_EDGE_PROPERTIES])})"
                session.execute(CREATE_EDGE_QUERY)
                create_flag = True

            if create_flag:
                # Wait for schema changes to propagate
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
            for tag_index in self.INIT_VERTEX_INDEXES:
                if not self._index_exists(session, self.space, "tag", tag_index['index_name']):
                    session.execute(f"CREATE TAG INDEX IF NOT EXISTS `{tag_index['index_name']}` ON `{self.INIT_VERTEX_TYPE}` ({tag_index['fields_str']})")
                    logger.info(f"Created tag index {tag_index['index_name']} on {self.INIT_VERTEX_TYPE}")
            for edge_index in self.INIT_EDGE_INDEXES:
                if not self._index_exists(session, self.space, "edge", edge_index['index_name']):
                    session.execute(f"CREATE EDGE INDEX IF NOT EXISTS `{edge_index['index_name']}` ON `{self.INIT_EDGE_TYPE}` ({edge_index['fields_str']})")
                    logger.info(f"Created edge index {edge_index['index_name']} on {self.INIT_EDGE_TYPE}")

            # Ensure Meta Knowledge Graph Indexes
            for tag_index in self.COMMUNITY_VERTEX_INDEXES:
                if not self._index_exists(session, self.space, "tag", tag_index['index_name']):
                    session.execute(f"CREATE TAG INDEX IF NOT EXISTS `{tag_index['index_name']}` ON `{self.COMMUNITY_VERTEX_TYPE}` ({tag_index['fields_str']})")
                    logger.info(f"Created tag index {tag_index['index_name']} on {self.COMMUNITY_VERTEX_TYPE}")

            # Wait for index creation to complete
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
        from ng_nx import NebulaReader, NebulaWriter
        from ng_nx.utils import NebulaGraphConfig

        self._init_nebulagraph_schema()

        self.config = NebulaGraphConfig(
            space=self.space,
            graphd_hosts=self.graphd_hosts,
            metad_hosts=self.metad_hosts
        )
        self.reader = NebulaReader(
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
        try:
            result = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]->(m) WHERE id(n) == $source_node_id AND id(m) == $target_node_id RETURN e",
                params={"source_node_id": source_node_id, "target_node_id": target_node_id}
            ).column_values("e")
            return len(result) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if edge exists between {source_node_id} and {target_node_id}: {e}") from e


    async def get_node(self, node_id: str) -> Union[dict, NodeView, None]:
        if node_id is None or not isinstance(node_id, str) or node_id == "":
            raise ValueError(f"Invalid node_id {node_id}")
        try:
            result = self.client.execute_py(
                f"MATCH (n:{self.INIT_VERTEX_TYPE}) WHERE id(n) == $node_id RETURN n",
                params={"node_id": node_id}
            ).column_values("n")
            return result[0] if result else None
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
            return result[0] if result else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get node degree for {node_id}: {e}") from e

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        # TODO: the implementation seems to be incorrect for NetworkX Storage:
        # return self._graph.degree(src_id) + self._graph.degree(tgt_id) <--- didn't look into MS Paper for this though, but this looks strange to me.
        if src_id is None or not isinstance(src_id, str) or src_id == "":
            raise ValueError(f"Invalid src_id {src_id}")
        if tgt_id is None or not isinstance(tgt_id, str) or tgt_id == "":
            raise ValueError(f"Invalid tgt_id {tgt_id}")
        try:
            result = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]->(m) WHERE id(n) == $src_id AND id(m) == $tgt_id RETURN count(e) AS Degree",
                params={"src_id": src_id, "tgt_id": tgt_id}
            ).column_values("Degree")
            return result[0] if result else 0
        except Exception as e:
            raise RuntimeError(f"Failed to get edge degree between {src_id} and {tgt_id}: {e}") from e

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        # TODO: Seems we should assume that src-->target is unique, this seems to be not true for either real-world or MS Paper.
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        if target_node_id is None or not isinstance(target_node_id, str) or target_node_id == "":
            raise ValueError(f"Invalid target_node_id {target_node_id}")
        try:
            result: list[dict] = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]->(m) WHERE id(n) == $src_id AND id(m) == $tgt_id RETURN e",
                params={"src_id": source_node_id, "tgt_id": target_node_id}
            ).as_primitive()
            # TODO: Now we return first edge, but may need revisit this.
            if not result:
                return None

            edge_primitive = result[0]
            if not edge_primitive or not edge_primitive.get('e'):
                return None
            edge = {
                'source_node_id': edge_primitive['e']['src'],
                'target_node_id': edge_primitive['e']['dst'],
                **edge_primitive['e']['props']
            }
            return edge

        except Exception as e:
            raise RuntimeError(f"Failed to get edge between {source_node_id} and {target_node_id}: {e}") from e

    async def get_node_edges(self, source_node_id: str) -> list[dict]:
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        try:
            result = self.client.execute_py(
                f"MATCH (n)-[e:{self.INIT_EDGE_TYPE}]->(m) WHERE id(n) == $src_id RETURN e",
                params={"src_id": source_node_id}
            ).as_primitive()

            if not result:
                return []
            edges = []
            for edge_primitive in result:
                if not edge_primitive or not edge_primitive.get('e'):
                    continue
                edge = {
                    'source_node_id': edge_primitive['e']['src'],
                    'target_node_id': edge_primitive['e']['dst'],
                    **edge_primitive['e']['props']
                }
                edges.append(edge)
            return edges
        except Exception as e:
            raise RuntimeError(f"Failed to get edges for node {source_node_id}: {e}") from e

    async def upsert_node(self, node_id: str, node_data: dict[str, str], label: Optional[str] = None):
        if node_id is None or not isinstance(node_id, str) or node_id == "":
            raise ValueError(f"Invalid node_id {node_id}")
        if node_data is None or not isinstance(node_data, dict):
            raise ValueError(f"Invalid node_data {node_data}")
        if not node_data:
            raise ValueError(f"Invalid node_data {node_data}")

        from uuid import uuid4
        label = label or self.INIT_VERTEX_TYPE

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
            f"  VALUES {node_id}:({prop_val});\n"
        )
        logger.debug(f"upsert_node()\nDML query: {query}")
        result = self.client.execute_py(query, props_map)
        # TODO: if prop missing in schema, try to alter schema to add the prop
        if not result.is_succeeded():
            raise RuntimeError(f"Failed to upsert node {node_id}: {result} with query {query}")

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str], label: Optional[str] = None):
        if source_node_id is None or not isinstance(source_node_id, str) or source_node_id == "":
            raise ValueError(f"Invalid source_node_id {source_node_id}")
        if target_node_id is None or not isinstance(target_node_id, str) or target_node_id == "":
            raise ValueError(f"Invalid target_node_id {target_node_id}")
        if edge_data is None or not isinstance(edge_data, dict):
            raise ValueError(f"Invalid edge_data {edge_data}")
        if not edge_data:
            raise ValueError(f"Invalid edge_data {edge_data}")

        from uuid import uuid4
        label = label or self.INIT_EDGE_TYPE

        prop_all_names = list(edge_data.keys())
        prop_name = ",".join(
            [prop for prop in prop_all_names if edge_data[prop] is not None]
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
        # TODO: consider add @RANK to enable multi-edge per src-->tgt
        query = (
            f"INSERT EDGE `{label}`({prop_name}) "
            f"  VALUES {source_node_id}->{target_node_id}:({prop_val});\n"
        )
        logger.debug(f"upsert_edge()\nDML query: {query}")
        result = self.client.execute_py(query, props_map)
        if not result.is_succeeded():
            raise RuntimeError(f"Failed to upsert edge between {source_node_id} and {target_node_id}: {result} with query {query}")


    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()


    def _cluster_data_to_graph(self, cluster_data: dict[str, list[dict[str, str]]]):
        # community node: (level, cluster), with id cluster_{cluster}
        # cluster_{cluster} is the key, and value is a list of (level, cluster)
        data: dict[str, list[int, int]] = defaultdict(list)
        # entity --> cluster
        # (entity_id, cluster_id, level)
        edges: list[tuple[int, int, int]] = []
        for node_id, clusters in cluster_data.items():
            for cluster in clusters:
                cluster_id = cluster["cluster"]
                level = cluster["level"]
                cluster_node_id = f'cluster_{cluster_id}'
                if cluster_node_id not in data:
                    data[cluster_node_id] = []
                data[cluster_node_id].append((level, cluster_id))
                edges.append((node_id, cluster_node_id, level))
        #    self._graph.nodes[node_id]["clusters"] = json.dumps(clusters) # We persist this into NebulaGraph

        # Write community data to NebulaGraph

        # Vertex data
        ng_nx_writer = self.writer_cls(
            data=data,
            nebula_config=self.config,
        )
        ng_nx_writer.set_options(
            label=self.COMMUNITY_VERTEX_TYPE,
            properties=["level", "cluster"],
            batch_size=self.BATCH_SIZE,
            write_mode="insert",
            sink="nebulagraph_vertex",
        )
        ng_nx_writer.write()
        # Edge data
        # TODO: add nebulagraph_edge sink for ng_nx_writer
        # https://github.com/wey-gu/NebulaGraph-nx/issues/6
        QUOTE = '"'
        for edge in edges:
            entity_id, cluster_node_id, level = edge
            # type check
            if not isinstance(level, int):
                raise ValueError(f"Invalid level {level} for edge {edge}")
            self.client.execute_py(
                f"INSERT EDGE `{self.COMMUNITY_EDGE_TYPE}`(level) "
                f"  VALUES {QUOTE}{entity_id}{QUOTE}->{QUOTE}{cluster_node_id}{QUOTE}:({level});\n"
            )
        return


    async def _leiden_clustering(self):
        from graspologic.partition import hierarchical_leiden

        # TOOD: introduce Cache mechanism for this.
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
        self._cluster_data_to_graph(node_communities)

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        # TODO:
        # 1. implement NebulaGraph KV Storage community report, findings, docs, chunks being stored as nodes and edges.
        # 2. implement community schema

        # Key: str(community_id)
        # Value: SingleCommunitySchema

        # SingleCommunitySchema = TypedDict(
        #     "SingleCommunitySchema",
        #     {
        #         "level": int,
        #         "title": str,
        #         "edges": list[list[str, str]],
        #         "nodes": list[str],
        #         "chunk_ids": list[str],
        #         "occurrence": float,
        #     },
        # )
        raise NotImplementedError("community_schema() is not implemented")


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
        # TODO: introduce cache mechnism, then we could leverage this callback
        pass



@dataclass
class StorageProfile:
    full_docs: Callable
    text_chunks: Callable
    llm_response_cache: Callable
    community_reports: Callable
    chunk_entity_relation: Callable
    entities: Callable

@dataclass
class StorageFactory:
    STORAGE_PROFILES: Dict[str, StorageProfile] = {
        "local": StorageProfile(
            full_docs=JsonKVStorage,
            text_chunks=JsonKVStorage,
            llm_response_cache=JsonKVStorage,
            community_reports=JsonKVStorage,
            chunk_entity_relation=NetworkXStorage,
            entities=MilvusLiteStorge
        ),
        "nebulagraph": StorageProfile(
            full_docs=NebulaGraphIndexStorage,
            text_chunks=NebulaGraphIndexStorage,
            llm_response_cache=JsonKVStorage,
            community_reports=NebulaGraphIndexStorage,
            chunk_entity_relation=NebulaGraphStorage,
            entities=MilvusLiteStorge
        )
    }

    @staticmethod
    def get_storage(
        namespace: str,
        global_config: Dict[str, Any],
        knowledge_store: Literal["local", "nebulagraph"] = "local",
        **kwargs
    ) -> Union[BaseKVStorage, BaseGraphStorage, BaseVectorStorage]:
        if knowledge_store not in StorageFactory.STORAGE_PROFILES:
            raise ValueError(f"Unsupported knowledge_store: {knowledge_store}")

        profile = StorageFactory.STORAGE_PROFILES[knowledge_store]

        storage_class = getattr(profile, namespace, None)
        if storage_class is None:
            raise ValueError(f"Unsupported namespace: {namespace}")

        return storage_class(namespace=namespace, global_config=global_config, **kwargs)
