import os
import fnmatch
import shutil
import tempfile
from typing import List

import zipfile
import io
import wget


from llama_index import Document, SimpleDirectoryReader, load_index_from_storage
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.node_parser import HierarchicalNodeParser
from loguru import logger
from llama_index.node_parser import get_leaf_nodes
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine


def find_files_with_extension(root_dir, extension):
    
    file_list = []
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            file_list.append(os.path.join(root, filename))
    return file_list
    

class AutoMergingSRDIndex:
    srd_folder_name = "srd"
    
    def __init__(self, llm, cache_dir="index/srd") -> None:
        self.cache_dir = "index/srd"
        self.llm = llm
        self.query_engine = None
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        srd_folder = self._compute_srd_dir_path(project_root)
        if not os.path.exists(srd_folder):
            srd_folder = self.download_srd(project_root)
        
        documents = self.load_srd_documents(srd_folder)
        
        # Extract the parser nodes and all the leaf nodes
        node_parser = self.build_node_parser()
        nodes = node_parser.get_nodes_from_documents(documents)
        self.storage_context = self.build_storage_context(nodes)
        
        if not os.path.exists(cache_dir):
            # If the cache directory does not exist, 
            # create the index from scratch

            # Convert the JSON files to LLaMaIndex documents
            logger.debug(f"Building index from {srd_folder}")
            

            leaf_nodes = get_leaf_nodes(nodes)

            service_context = self.build_service_context()
            
            # Build the index
            automerging_index = VectorStoreIndex(
                leaf_nodes, 
                storage_context=self.storage_context, 
                service_context=service_context
            )
            self.index = automerging_index
            
            os.makedirs(self.cache_dir, exist_ok=True)
            self.store_index(automerging_index)
                
        else:
            # If the cache directory exists, load the index from disk
            self.index = self.load_index(self.storage_context)
        
        # The query engine is used to search the index
        self.query_engine = self.get_automerging_query_engine(self.index)

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
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
        
        return node_parser
    
    def build_storage_context(self, documents) -> StorageContext:
        """Build the storage context for the index."""
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(documents)
        
        return storage_context
                
    def build_service_context(self, embed_model="local:BAAI/bge-small-en-v1.5") -> ServiceContext:
        """Build the service context for the index."""
        return ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=embed_model,
            node_parser=self.build_node_parser(),
        )
        
    def download_srd(self, destination_path) -> str:
        """Download the SRD from the Dropbox link and unzip it."""
        
        return self._compute_srd_dir_path(destination_path)
    
        # TODO DropBox link is not working properly

        file_path = os.path.join(destination_path, "srd.zip")
        SRD_URL = "https://www.dropbox.com/scl/fi/h6zfhincgxwydijlf8ixk/srd.zip?rlkey=u9yu5r1ly1273st5fu84664ag&dl=0"
        
        # os.system(f"wget -qq -O {file_path} {SRD_URL}")

        # Unzip the file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_path)
                
        # Remove the zip file
        shutil.rmtree(file_path)
        
        return self._compute_srd_dir_path(destination_path)
        
    
    def load_index(self, context) -> VectorStoreIndex:
        """Load the index from disk."""
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=self.cache_dir),
            service_context=context
        )
        return index
    
    def store_index(self, index):
        """Store the index to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=self.cache_dir)
    
    def get_automerging_query_engine(
        self,
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=6,
    ) -> RetrieverQueryEngine:
        base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(
            base_retriever, automerging_index.storage_context, verbose=True
        )
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[rerank]
        )
        return auto_merging_engine
    
    def query(self, query: str):
        """Query the index."""
        return self.query_engine.query(query)
    
if __name__ == "__main__":
    from llama_index.llms import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    index = AutoMergingSRDIndex(llm)
    
    print(index.query("What is the attack bonus for a longsword?"))
    
    