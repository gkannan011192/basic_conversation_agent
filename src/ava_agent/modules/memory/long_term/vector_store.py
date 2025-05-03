import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List,Optional
from ava_agent.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,PointStruct,VectorParams
from sentence_transformers import SentenceTransformer

@dataclass
class Memory:
    """A class representing a memory entry with text, metadata, and score.
    Attributes:
        text (str): The text content of the memory.
        metadata (dict): Dictionary containing metadata about the memory.
        score (float, optional): A score associated with the memory. Defaults to None.
    Properties:
        id (str, optional): The unique identifier of the memory from metadata.
        timestamp (datetime, optional): The timestamp of the memory from metadata, 
            parsed from ISO format string.
    """
    text: str
    metadata: dict
    score: Optional[float]=None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")
    @property
    def timestamp(self) -> Optional[datetime]:
        ts=self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None

class VectorStore:
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9  # Threshold for considering memories as similar

    _instance: Optional["VectorStore"]=None
    _initialized: bool=False

    def __new__(cls) -> "VectorStore":
        if not cls._instance:
            cls._instance=super().__new__(cls)
        return cls._instance
    def __init__(self) -> None:
        if not self._initialized:
            # self._validate_env_variables()
            self.model=SentenceTransformer(
                self.EMBEDDING_MODEL
            )
            self.client=QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            self._initialized=True
        
    def _validate_env_variables(self) -> None:
        """
        Validates the presence of required environment variables.
        This method checks if all the required environment variables defined in
        REQUIRED_ENV_VARS are set. If any variables are missing, it raises a ValueError
        with details of the missing variables.
        Raises:
            ValueError: If one or more required environment variables are not set.
                The error message includes a comma-separated list of missing variables.
        """

        missing_variables=[var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_variables:
            raise ValueError("Missing required environment variables: {}".format(",".join(missing_variables)))

    def _collection_exists(self) -> bool:
        """
        Check if the collection exists in the vector store.
        This method verifies if a collection with the name specified by self.COLLECTION_NAME
        exists in the Qdrant client's database.
        Returns:
            bool: True if the collection exists, False otherwise.
        """

        collections=self.client.get_collections().collections
        return any(collection.name==self.COLLECTION_NAME for collection in collections)
    
    def _create_collection(self) -> None:
        """
        Creates a new collection in the vector database if it doesn't exist.
        The collection is initialized with vector parameters determined by encoding a sample text.
        The embedding dimension is inferred from the sample encoding, and cosine similarity is used
        as the distance metric.
        This is an internal helper method used during initialization of the vector store.
        Requires:
            - self.model: An encoding model that can convert text to vectors
            - self.client: A database client that supports collection creation
            - self.COLLECTION_NAME: Name of the collection to create
        Returns:
            None
        """

        sample_embedding=self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE
            )
        )
    
    def find_similar_memory(self,text:str) -> Optional[Memory]:
        """
        Search for a memory similar to the given text and return if found above threshold.
        Args:
            text (str): Text to search for similar memories
        Returns:
            Optional[Memory]: Most similar memory if found above threshold, None otherwise
        Note:
            Uses configured similarity threshold to determine if memory is similar enough
        """

        search_memory_results=self.search_memories(text,k=1)
        if search_memory_results and search_memory_results[0].score>=self.SIMILARITY_THRESHOLD:
            return search_memory_results[0]
        return None

    def store_memory(self,text:str, metadata: dict) -> None:
        """
        Store a memory text with associated metadata in the vector store.
        This method stores or updates a text and its metadata in the vector store. If a similar memory
        exists, it will update that memory instead of creating a new one. The text is encoded into
        a vector embedding before storage.
        Args:
            text (str): The text content to store as a memory
            metadata (dict): Additional metadata to store with the text
        Returns:
            None
        Example:
            >>> memory_store.store_memory(
            ...     "The sky is blue",
            ...     {"timestamp": "2023-01-01", "source": "observation"}
            ... )
        """

        if not self._collection_exists():
            self._create_collection()
        similar_memory=self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"]=similar_memory.id
        embedding= self.model.encode(text)
        vector_point=PointStruct(
            id=metadata.get("id",hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[vector_point]
        )

    def search_memories(self,query: str, k:int=5) -> List[Memory]:
        """
        Search for memories in the vector store based on a query string.
        This method performs a semantic search using vector embeddings to find relevant memories
        matching the input query. If the collection does not exist, returns an empty list.
        Args:
            query (str): The search query text to find relevant memories
            k (int, optional): Maximum number of memories to return. Defaults to 5.
        Returns:
            List[Memory]: A list of Memory objects containing the search results, sorted by relevance.
                Each Memory object includes:
                - text: The memory content
                - metadata: Associated metadata excluding the text field
                - score: Similarity score of the match
        Example:
            >>> memories = vector_store.search_memories("meeting with John", k=3)
        """

        if not self._collection_exists():
            return []
        query_embedding=self.model.encode(query)
        search_results=self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        return [
            Memory(
                text=result.payload["text"],
                metadata={key:value for key,value in result.payload.items() if key!="text"},
                score=result.score
            )
            for result in search_results
        ]
    
@lru_cache
def get_vector_store() -> VectorStore:
    """
    Returns a singleton vector store
    """
    return VectorStore()
    

        
    
     