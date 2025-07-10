from typing import List, Dict, Optional, Union
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


class QdrantVectorStore:
    def __init__(self, uri: str):
        """
        Initialize a connection to Qdrant server.
        :param uri: URL to the Qdrant server (e.g., 'http://localhost:6333')
        """
        self.uri = uri
        self.client = self._connect_to_qdrant()

    def _connect_to_qdrant(self) -> QdrantClient:
        """Establish and return a Qdrant client connection."""
        try:
            return QdrantClient(url=self.uri)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant at {self.uri}: {e}")

    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        distance: Union[str, models.Distance] = "cosine",
    ):
        """
        Create a new collection in Qdrant.

        :param collection_name: Name of the collection.
        :param embedding_size: Dimension of the embedding vector.
        :param distance: Distance metric to use ('cosine', 'euclid', 'dot', 'manhattan' or models.Distance).
        """
        # Convert string to models.Distance if needed
        if isinstance(distance, str):
            distance = distance.lower()
            if distance == "cosine":
                distance = models.Distance.COSINE
            elif distance == "euclid":
                distance = models.Distance.EUCLID
            elif distance == "dot":
                distance = models.Distance.DOT
            elif distance == "manhattan":
                distance = models.Distance.MANHATTAN
            else:
                raise ValueError(f"Unsupported distance metric: '{distance}'")

        try:
            if not self.client.collection_exists(collection_name=collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=distance,
                    ),
                    on_disk_payload=True,
                )
        except UnexpectedResponse as e:
            raise RuntimeError(f"Failed to create collection '{collection_name}': {e}")

    def add_embeddings(
        self,
        collection_name: str,
        embeddings: List[np.ndarray],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[Union[str, int]]] = None,
    ):
        """
        Add multiple embeddings to a collection.

        :param collection_name: Target Qdrant collection.
        :param embeddings: List of numpy arrays (each is a vector).
        :param payloads: Optional list of metadata dictionaries.
        :param ids: Optional list of custom point IDs (UUIDs will be generated if not provided).
        """
        if payloads and len(payloads) != len(embeddings):
            raise ValueError("Payload list must match the length of embeddings")

        if ids and len(ids) != len(embeddings):
            raise ValueError("IDs list must match the length of embeddings")

        points = []
        for i, embedding in enumerate(embeddings):
            vector = embedding.tolist()
            payload = payloads[i] if payloads else None
            point_id = ids[i] if ids else str(uuid4())

            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def query(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter: Optional[models.Filter] = None,
    ) -> List[models.ScoredPoint]:
        """
        Query for similar vectors.

        :param collection_name: Target collection.
        :param query_vector: The input vector to search with.
        :param top_k: Number of results to return.
        :param filter: Optional Qdrant filter for conditional queries.
        :return: List of scored points with IDs, scores, and optional payloads.
        """
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=filter,
        )
        return search_result
