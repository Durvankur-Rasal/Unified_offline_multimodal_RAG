import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# IMPORT YOUR DP WRAPPER
from src.dp_embeddings import DPHuggingFaceEmbeddings


class RAGPipeline:
    def __init__(
        self,
        model_path: str = "models/meta-llama-3.1-8b-instruct-q4_k_m.gguf",
        index_dir: str = "faissindex",
    ):
        self.model_path = model_path
        self.index_dir = index_dir

        # -----------------------------
        # Load Embeddings
        # -----------------------------
        print("Loading Differentially Private Embedding Model...")
        self.embeddings = DPHuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            epsilon=1.2,
        )

        # -----------------------------
        # Load FAISS Vector Store
        # -----------------------------
        print("Loading local FAISS vector store...")
        if not os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_dir}/. Did you run ingest.py?"
            )

        self.vectorstore = FAISS.load_local(
            self.index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # -----------------------------
        # Load Local LLM (LlamaCpp)
        # -----------------------------
        print(f"Loading local LLM from {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LLM model not found at {self.model_path}."
            )

        self.llm = LlamaCpp(
           model_path=self.model_path,
            temperature=0.1,
            max_tokens=512,
            n_ctx=3072,           # ⬇️ REDUCED from 4096 to save massive memory
            n_threads=8,          # ⚡ ADD THIS: Change '8' to match your CPU's core count
            n_batch=512,          # ⚡ ADD THIS: Speeds up prompt processing
            n_gpu_layers=20,      # ⚡ ADD THIS: Offloads math to your GPU. (Use 0 if you don't have a GPU)
            echo=False,
            stop=["<|eot_id|>", "<|eom_id|>"]
        )

        # -----------------------------
        # Prompt Template (UPDATED)
        # -----------------------------
        # EXACT Llama 3.1 Instruct Prompt Format tailored for Healthcare
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are a highly secure, offline Clinical AI Assistant. Your purpose is to assist medical professionals by retrieving patient data and medical guidelines.
            STRICT RULES:
            1. ONLY answer using the information from the provided CONTEXT. 
            2. If the context does not contain the answer, explicitly state: "Insufficient patient data in offline records."
            3. Never invent or hallucinate medical readings, dosages, or diagnoses.
            4. Maintain a professional, clinical tone.

            CONTEXT START
            {context}
            CONTEXT END<|eot_id|><|start_header_id|>user<|end_header_id|>

            {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input"],
        )

        # -----------------------------
        # Helper: Format Documents
        # -----------------------------
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # -----------------------------
        # Build RAG Pipeline (Runnable)
        # -----------------------------
        print("Building RAG pipeline (LangChain v1)...")

        self.qa_chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "input": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

    # -----------------------------
    # Ask Query
    # -----------------------------
    def ask(self, query: str) -> dict:
        # Retrieve documents separately (for returning sources)
        docs = self.retriever.invoke(query)

        # Generate answer
        answer = self.qa_chain.invoke(query)

        return {
            "result": answer,
            "source_documents": docs,
        }