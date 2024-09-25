import json
import asyncio
from collections import defaultdict
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union
from ..base import BaseGraphStorage, SingleCommunitySchema
from .._utils import logger
from ..prompt import GRAPH_FIELD_SEP

neo4j_lock = asyncio.Lock()


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_")


@dataclass
class Neo4jStorage(BaseGraphStorage):
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url, auth=self.neo4j_auth
        )

    # async def create_database(self):
    #     async with self.async_driver.session() as session:
    #         try:
    #             constraints = await session.run("SHOW CONSTRAINTS")
    #             # TODO I don't know why CREATE CONSTRAINT IF NOT EXISTS still trigger error
    #             # so have to check if the constrain exists
    #             constrain_exists = False

    #             async for record in constraints:
    #                 if (
    #                     self.namespace in record["labelsOrTypes"]
    #                     and "id" in record["properties"]
    #                     and record["type"] == "UNIQUENESS"
    #                 ):
    #                     constrain_exists = True
    #                     break
    #             if not constrain_exists:
    #                 await session.run(
    #                     f"CREATE CONSTRAINT FOR (n:{self.namespace}) REQUIRE n.id IS UNIQUE"
    #                 )
    #                 logger.info(f"Add constraint for namespace: {self.namespace}")

    #         except Exception as e:
    #             logger.error(f"Error accessing or setting up the database: {str(e)}")
    #             raise

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()
        # TODOLater: create database if not exists always cause an error when async
        # await self.create_database()

    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN COUNT(r) > 0 AS exists",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id "
                f"RETURN COUNT {{(n)-[]-(:{self.namespace})}} AS degree",
                node_id=node_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $src_id AND t.id = $tgt_id "
                f"RETURN COUNT {{(s)-[]-(:{self.namespace})}} + COUNT {{(t)-[]-(:{self.namespace})}} AS degree",
                src_id=src_id,
                tgt_id=tgt_id,
            )
            record = await result.single()
            return record["degree"] if record else 0

    async def get_node(self, node_id: str) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:{self.namespace}) WHERE n.id = $node_id RETURN properties(n) AS node_data",
                node_id=node_id,
            )
            record = await result.single()
            raw_node_data = record["node_data"] if record else None
        if raw_node_data is None:
            return None
        raw_node_data["clusters"] = json.dumps(
            [
                {
                    "level": index,
                    "cluster": cluster_id,
                }
                for index, cluster_id in enumerate(
                    raw_node_data.get("communityIds", [])
                )
            ]
        )
        return raw_node_data

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "RETURN properties(r) AS edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
            )
            record = await result.single()
            return record["edge_data"] if record else None

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (s:{self.namespace})-[r]->(t:{self.namespace}) WHERE s.id = $source_id "
                "RETURN s.id AS source, t.id AS target",
                source_id=source_node_id,
            )
            edges = []
            async for record in result:
                edges.append((record["source"], record["target"]))
            return edges

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
        async with self.async_driver.session() as session:
            await session.run(
                f"MERGE (n:{self.namespace}:{node_type} {{id: $node_id}}) "
                "SET n += $node_data",
                node_id=node_id,
                node_data=node_data,
            )

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        edge_data.setdefault("weight", 0.0)
        async with self.async_driver.session() as session:
            await session.run(
                f"MATCH (s:{self.namespace}), (t:{self.namespace}) "
                "WHERE s.id = $source_id AND t.id = $target_id "
                "MERGE (s)-[r:RELATED]->(t) "  # Added relationship type 'RELATED'
                "SET r += $edge_data",
                source_id=source_node_id,
                target_id=target_node_id,
                edge_data=edge_data,
            )

    async def clustering(self, algorithm: str):
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session() as session:
            try:
                # Project the graph with undirected relationships
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: True,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                community_count: int = result["communityCount"]
                modularities = result["modularities"]
                logger.info(
                    f"Performed graph clustering with {community_count} communities and modularities {modularities}"
                )
            finally:
                # Drop the projected graph
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")

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

        async with self.async_driver.session() as session:
            # Fetch community data
            result = await session.run(
                f"""
                MATCH (n:{self.namespace})
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:{self.namespace}) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            # records = await result.fetch()

            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session() as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:{self.namespace})-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:{self.namespace}) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
