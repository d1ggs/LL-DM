import fnmatch
import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.llama_cpp import LlamaCPP
from loguru import logger


def find_files_with_extension(root_dir, extension):

    file_list = []
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f"*.{extension}"):
            file_list.append(os.path.join(root, filename))
    return file_list


@dataclass
class SRDConfig:
    srd_folder_path: str = "srd"
    index_cache_path: str = "index"


class AutoMergingSRDIndex:
    srd_folder_name = "srd"
    embedding_model = "local:BAAI/bge-small-en-v1.5"

    def __init__(self, llm: LlamaCPP, config: SRDConfig) -> None:
        self.cache_dir = config.index_cache_path
        self.llm = llm
        self.query_engine = None

        if not os.path.exists(self.cache_dir):
            logger.debug(f"Creating index from scratch in {self.cache_dir}")

            srd_folder = config.srd_folder_path

            logger.debug(f"Loading SRD documents from {srd_folder}")
            documents = self.load_srd_documents(srd_folder)

            # Extract the parser nodes and all the leaf nodes
            node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128]
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            leaf_nodes = get_leaf_nodes(nodes)

            docstore = SimpleDocumentStore()
            # insert nodes into docstore
            docstore.add_documents(nodes)
            # define storage context (will include vector store by default too)
            storage_context = StorageContext.from_defaults(docstore=docstore)
            self.index = VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                embed_model=self.embedding_model,
                show_progress=True,
            )
            self.store_index()
        else:
            logger.debug(f"Cache directory {self.cache_dir} already exists")
            logger.debug(f"Trying to load existing index from {self.cache_dir}")
            self.load_index()

        logger.debug("Index ready")
        
        logger.debug("Building query engine")
        # The query engine is used to search the index
        self.query_engine = self.get_automerging_query_engine(self.index)
        logger.debug("Query engine ready")
        
    def _compute_srd_dir_path(self, project_root):
        return os.path.join(project_root, self.srd_folder_name)

    def load_srd_documents(self, srd_dir: str) -> List[Document]:
        """Load the SRD documents from the given directory."""
        input_files = find_files_with_extension(srd_dir, "json")
        logger.debug(f"Found {len(input_files)} files in {srd_dir}")

        # Remove the index.json file from the list of input files
        input_files = [f for f in input_files if "index.json" not in f]
        documents = SimpleDirectoryReader(input_files=input_files).load_data()

        return documents

    def build_node_parser(self, chunk_sizes=[2048, 512, 128]) -> HierarchicalNodeParser:
        """Build the node parser for the index."""
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])

        return node_parser

    def build_storage_context(self, persist_dir: str) -> StorageContext:
        """Build the storage context for the index.

        Parameters
        ----------

        persist_dir: str
            The directory where the index will be stored.

        Returns
        -------

        storage_context: StorageContext
            The storage context for the index.

        """

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

        return storage_context

    def download_srd(self, destination_path) -> str:
        """Download the SRD from the Dropbox link and unzip it."""

        return self._compute_srd_dir_path(destination_path)

        # TODO DropBox link is not working properly

        file_path = os.path.join(destination_path, "srd.zip")
        SRD_URL = "https://www.dropbox.com/scl/fi/h6zfhincgxwydijlf8ixk/srd.zip?rlkey=u9yu5r1ly1273st5fu84664ag&dl=0"

        # os.system(f"wget -qq -O {file_path} {SRD_URL}")

        # Unzip the file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(destination_path)

        # Remove the zip file
        shutil.rmtree(file_path)

        return self._compute_srd_dir_path(destination_path)

    def load_index(self):
        """Load the index from disk and set the self.index attribute."""

        # rebuild storage context
        storage_context = self.build_storage_context(persist_dir=self.cache_dir)

        # load index
        index = load_index_from_storage(
            storage_context, embed_model=self.embedding_model, show_progress=True
        )

        self.index = index

    def store_index(self):
        """Store the index to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.cache_dir)

    def get_automerging_query_engine(
        self,
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=6,
    ) -> RetrieverQueryEngine:

        base_retriever = automerging_index.as_retriever(
            similarity_top_k=similarity_top_k
        )

        retriever = AutoMergingRetriever(
            base_retriever, automerging_index.storage_context, verbose=True
        )
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, 
            node_postprocessors=[rerank], 
            llm=self.llm, 
            response_mode="tree_summarize", 
            similarity_top_k=20
        )
        return auto_merging_engine

    def query(self, query: str):
        """Query the index."""
        return self.query_engine.query(query)


if __name__ == "__main__":
    from llama_index.llms import OpenAI

    config = SRDConfig(srd_folder_path="srd", index_cache_path="index")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    index = AutoMergingSRDIndex(llm, config)

    print(index.query("What is the attack bonus for a longsword?"))
