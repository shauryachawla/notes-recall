from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
import openai
import logging

# load_dotenv()

MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,  # Default for 3-small is 1536, but can be reduced
        "text-embedding-3-large": 3072,  # Default for 3-large is 3072, but can be reduced
        "text-embedding-ada-002": 1536,  # Fixed dimensions
    }

class RAG:
    """A class for Retrieval-Augmented Generation using OpenAI embeddings."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dimension = MODEL_DIMENSIONS.get(model_name, "text-embedding-3-small")
        if self.embedding_dimension is None:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""])

        self.client = openai.OpenAI()
        logging.info(
            "Initialized OpenAI provider with model=%s, dimensions=%s",
            self.model_name,
            self.embedding_dimension
        )

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using the configured text splitter."""
        return self.text_splitter.split_text(text)

    def get_embedding(self, text: str) -> list[float]:
        """Get the embedding for a given text using the OpenAI API."""
        text = text.replace("\n", " ")  # Ensure the text is a single line
        response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
        return response.data[0].embedding


__all__ = ["RAG"]
