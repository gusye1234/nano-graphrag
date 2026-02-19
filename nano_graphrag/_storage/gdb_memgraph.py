import json
import asyncio
from collections import defaultdict
from typing import List
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union
from ..base import BaseGraphStorage, SingleCommunitySchema
from .._utils import logger
from ..prompt import GRAPH_FIELD_SEP

memgraph_lock = asyncio.Lock()


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_").replace(":", "_").replace("\\", "__")


@dataclass
class MemgraphStorage(BaseGraphStorage):
    def __post_init__(self):
        self.memgraph_url = self.global_config["addon_params"].get("memgraph_url", None)
        self.memgraph_auth = self.global_config["addon_params"].get("memgraph_auth", None)
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        logger.info(f"Using the label {self.namespace} for Memgraph as identifier")
        if self.memgraph_url is None:
            raise ValueError("Missing memgraph_url in addon_params")
        self.async_driver = AsyncGraphDatabase.driver(
            self.memgraph_url, auth=self.memgraph_auth, max_connection_pool_size=50,      
        )

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()

    async def index_start_callback(self):
        logger.info("Init Memgraph workspace")
        await self._init_workspace()
        
        # create index for faster searching
        try:
            async with self.async_driver.session() as session:
                # Memgraph uses CREATE INDEX ON syntax
                await session.run(
                    f"CREATE INDEX ON :`{self.namespace}`(id);"
                )
                
                await session.run(
                    f"CREATE INDEX ON :`{self.namespace}`(entity_type);"
                )
                
                await session.run(
                    f"CREATE INDEX ON :`{self.namespace}`(communityIds);"
                )
                
                await session.run(
                    f"CREATE INDEX ON :`{self.namespace}`(source_id);"
                )          
                logger.info("Memgraph indexes created successfully")                
        except Exception as e:
            # Index might already exist, log warning but don't fail
            logger.warning(f"Index creation warning (might already exist): {e}")

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"MATCH (n:`{self.namespace}`) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                MATCH (s:`{self.namespace}`)-[r:RELATED]->(t:`{self.namespace}`)
                WHERE s.id = $source_id AND t.id = $target_id
                RETURN COUNT(r) > 0 AS exists
                """,
                source_id=source_node_id,
                target_id=target_node_id,
            )
    
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        results = await self.node_degrees_batch([node_id])
        return results[0] if results else 0
        
    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        if not node_ids:
            return {}
                    
        result_dict = {node_id: 0 for node_id in node_ids}
        async with self.async_driver.session() as session:
            result = await session.run(
                f"""
                UNWIND $node_ids AS node_id
                MATCH (n:`{self.namespace}`)
                WHERE n.id = node_id
                OPTIONAL MATCH (n)-[]-(m:`{self.namespace}`)
                RETURN node_id, COUNT(m) AS degree
                """,
                node_ids=node_ids
            )
                
            async for record in result:
                result_dict[record["node_id"]] = record["degree"]
                
        return [result_dict[node_id] for node_id in node_ids]
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        results = await self.edge_degrees_batch([(src_id, tgt_id)])
        return results[0] if results else 0

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        if not edge_pairs:
            return []
        
        result_dict = {tuple(edge_pair): 0 for edge_pair in edge_pairs}
        
        edges_params = [{"src_id": src, "tgt_id": tgt} for src, tgt in edge_pairs]
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    
                    MATCH (s:`{self.namespace}`)
                    WHERE s.id = edge.src_id
                    WITH edge, s
                    OPTIONAL MATCH (s)-[]-(n1:`{self.namespace}`)
                    WITH edge, COUNT(n1) AS src_degree
                    
                    MATCH (t:`{self.namespace}`)
                    WHERE t.id = edge.tgt_id
                    WITH edge, src_degree, t
                    OPTIONAL MATCH (t)-[]-(n2:`{self.namespace}`)
                    WITH edge.src_id AS src_id, edge.tgt_id AS tgt_id, src_degree, COUNT(n2) AS tgt_degree
                    
                    RETURN src_id, tgt_id, src_degree + tgt_degree AS degree
                    """,
                    edges=edges_params
                )
                
                async for record in result:
                    result_dict[(record["src_id"], record["tgt_id"])] = record["degree"]
            
            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge degree calculation: {e}")
            return [0] * len(edge_pairs)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        result = await self.get_nodes_batch([node_id])
        return result[0] if result else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        if not node_ids:
            return {}
            
        result_dict = {node_id: None for node_id in node_ids}

        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (n:`{self.namespace}`)
                    WHERE n.id = node_id
                    RETURN node_id, properties(n) AS node_data
                    """,
                    node_ids=node_ids
                )
                
                async for record in result:
                    result_dict[record["node_id"]] = record["node_data"]
                    
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node retrieval: {e}")
            raise e

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        results = await self.get_edges_batch([(source_node_id, target_node_id)])
        return results[0] if results else None

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        if not edge_pairs:
            return []
            
        result_dict = {tuple(edge_pair): None for edge_pair in edge_pairs}
        
        edges_params = [{"source_id": src, "target_id": tgt} for src, tgt in edge_pairs]
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    MATCH (s:`{self.namespace}`)-[r:RELATED]->(t:`{self.namespace}`)
                    WHERE s.id = edge.source_id AND t.id = edge.target_id
                    RETURN edge.source_id AS source_id, edge.target_id AS target_id, properties(r) AS edge_data
                    """,
                    edges=edges_params
                )
                
                async for record in result:
                    result_dict[(record["source_id"], record["target_id"])] = record["edge_data"]
            
            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge retrieval: {e}")
            return [None] * len(edge_pairs)

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]]:
        results = await self.get_nodes_edges_batch([source_node_id])
        return results[0] if results else []

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        if not node_ids:
            return []
            
        result_dict = {node_id: [] for node_id in node_ids}
        
        try:
            async with self.async_driver.session() as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (s:`{self.namespace}`)-[r:RELATED]->(t:`{self.namespace}`)
                    WHERE s.id = node_id
                    RETURN node_id, s.id AS source_id, t.id AS target_id
                    """,
                    node_ids=node_ids
                )
                
                async for record in result:
                    result_dict[record["node_id"]].append((record["source_id"], record["target_id"]))
            
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node edges retrieval: {e}")
            return [[] for _ in node_ids]

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        await self.upsert_nodes_batch([(node_id, node_data)])

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        if not nodes_data:
            return []
        
        nodes_by_type = {}
        for node_id, node_data in nodes_data:
            node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_data))
        
        async with self.async_driver.session() as session:
            for node_type, type_nodes in nodes_by_type.items():
                nodes_params = []
                for node_id, node_data in type_nodes:
                    node_data_copy = node_data.copy()
                    node_data_copy["id"] = node_id
                    node_data_copy.setdefault("clusters", "[]")
                    nodes_params.append(node_data_copy)
                
                await session.run(
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:`{self.namespace}` {{id: node.id}})
                    SET n += node, n:`{node_type}`
                    """,
                    nodes=nodes_params
                )
        
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])

    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        if not edges_data:
            return
        
        edges_params = []
        for source_id, target_id, edge_data in edges_data:
            edge_data_copy = edge_data.copy() 
            edge_data_copy.setdefault("weight", 0.0)
            
            edges_params.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_data": edge_data_copy
            })
        
        async with self.async_driver.session() as session:
            await session.run(
                f"""
                UNWIND $edges AS edge
                MATCH (s:`{self.namespace}`)
                WHERE s.id = edge.source_id
                WITH edge, s
                MATCH (t:`{self.namespace}`)
                WHERE t.id = edge.target_id
                MERGE (s)-[r:RELATED]->(t)
                SET r += edge.edge_data
                """,
                edges=edges_params
            )

    async def clustering(self, algorithm: str):
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Memgraph implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session() as session:
            try:
                # First try using Memgraph MAGE (if available)
                try:
                    # Create a subgraph for clustering - Memgraph MAGE syntax
                    await session.run(
                        f"""
                        CALL graph.create(
                            "clustering_graph",
                            "MATCH (n:`{self.namespace}`) RETURN n",
                            "MATCH (n:`{self.namespace}`)-[r:RELATED]->(m:`{self.namespace}`) RETURN r"
                        );
                        """
                    )
                    
                    # Run Leiden clustering using Memgraph MAGE
                    result = await session.run(
                        f"""
                        CALL community_detection.leiden("clustering_graph") 
                        YIELD node, community_id
                        RETURN node, community_id;
                        """
                    )
                    
                    # Process clustering results
                    community_map = {}
                    async for record in result:
                        node_id = record["node"]["id"]
                        community_id = record["community_id"]
                        community_map[node_id] = community_id
                    
                    # Update nodes with community information
                    for node_id, community_id in community_map.items():
                        await session.run(
                            f"""
                            MATCH (n:`{self.namespace}`)
                            WHERE n.id = $node_id
                            SET n.communityIds = $community_id
                            """,
                            node_id=node_id,
                            community_id=str(community_id)
                        )
                    
                    logger.info("Successfully completed Leiden clustering using Memgraph MAGE")
                    
                except Exception as mage_error:
                    logger.warning(f"MAGE clustering failed: {mage_error}")
                    logger.info("Falling back to simple connected components clustering")
                    
                    # Fallback: Use connected components (simpler clustering)
                    result = await session.run(
                        f"""
                        MATCH (n:`{self.namespace}`)
                        OPTIONAL MATCH path=(n)-[:RELATED*]-(m:`{self.namespace}`)
                        WITH n, COLLECT(DISTINCT CASE WHEN m IS NOT NULL THEN m.id ELSE n.id END) AS component
                        WITH n, component, SIZE(component) AS component_size
                        SET n.communityIds = toString(ABS(HASH(component[0])))
                        RETURN n.id AS node_id, n.communityIds AS community_id
                        """
                    )
                    
                    community_count = 0
                    async for record in result:
                        community_count += 1
                    
                    logger.info(f"Completed fallback clustering with {community_count} nodes assigned")
                
            except Exception as e:
                logger.error(f"Error during clustering: {e}")
                # Final fallback: assign all nodes to community 0
                await session.run(
                    f"""
                    MATCH (n:`{self.namespace}`)
                    SET n.communityIds = "0"
                    """
                )
                logger.warning("Used final fallback: assigned all nodes to community 0")
            finally:
                # Clean up the subgraph if it was created
                try:
                    await session.run("CALL graph.drop('clustering_graph');")
                except:
                    pass  # Graph might not exist or MAGE might not be available

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
                MATCH (n:`{self.namespace}`)
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:`{self.namespace}`) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            max_num_ids = 0
            async for record in result:
                node_id = record["node_id"]
                source_id = record["source_id"]
                cluster_key = record["cluster_key"]
                connected_nodes = record["connected_nodes"]

                if cluster_key:
                    try:
                        cluster_ids = json.loads(cluster_key) if isinstance(cluster_key, str) else [cluster_key]
                        max_num_ids = max(max_num_ids, len(cluster_ids))
                        
                        for i, cluster_id in enumerate(cluster_ids):
                            cluster_key = f"{cluster_id}_{i}"
                            results[cluster_key]["level"] = i
                            results[cluster_key]["title"] = f"Cluster {cluster_id} Level {i}"
                            results[cluster_key]["nodes"].add(node_id)
                            if source_id:
                                results[cluster_key]["chunk_ids"].add(source_id)
                            
                            # Add edges between connected nodes in the same cluster
                            for connected_node in connected_nodes:
                                if connected_node != node_id:
                                    edge = tuple(sorted([node_id, connected_node]))
                                    results[cluster_key]["edges"].add(edge)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback for non-JSON cluster keys
                        cluster_key = f"{cluster_key}_0"
                        results[cluster_key]["level"] = 0
                        results[cluster_key]["title"] = f"Cluster {cluster_key} Level 0"
                        results[cluster_key]["nodes"].add(node_id)
                        if source_id:
                            results[cluster_key]["chunk_ids"].add(source_id)

            # Process results
            for k, v in results.items():
                v["edges"] = list(v["edges"])
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"])

            # Compute sub-communities (this is a simplified approach)
            for cluster in results.values():
                cluster["sub_communities"] = []

        return dict(results)

    async def index_done_callback(self):
        await self.async_driver.close()

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session() as session:
            try:
                await session.run(f"MATCH (n:`{self.namespace}`) DETACH DELETE n")
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {e}")
                raise e
