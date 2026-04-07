import os
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# IMPORT OUR DP WRAPPER
from src.dp_embeddings import DPHuggingFaceEmbeddings

# 1. DEFINE THE AGENT'S MEMORY (STATE)
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    context: Optional[str]
    math_result: Optional[str]
    final_answer: Optional[str]

class AgenticRAG:
    def __init__(self, index_dir: str = "faissindex"):
        self.index_dir = index_dir
        
        print("Loading Differentially Private Embeddings...")
        self.embeddings = DPHuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            epsilon=1.2 
        )
        
        print("Loading local FAISS vector store...")
        if not os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            raise FileNotFoundError("FAISS index not found! Run ingest.py first.")
            
        self.vectorstore = FAISS.load_local(
            self.index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        print("Connecting Agent Brain to Ollama (Phi-3)...")
        self.llm = Ollama(
            model="phi3",
            temperature=0.0, # 0.0 makes the agent strictly logical for routing
            num_ctx=2048,
            stop=["<|eot_id|>"]
        )

        # Build the LangGraph
        self.app = self._build_graph()

    # --- NODE 1: THE ROUTER ---
    def _route_query(self, state: AgentState) -> AgentState:
        print("\n[AGENT] Analyzing Intent...")
        
        route_prompt = PromptTemplate.from_template("""
        You are a clinical routing assistant. Read the following medical query and classify it into one of two categories:
        1. 'search' - if the query asks for patient history, symptoms, medical records, or guidelines.
        2. 'calculate' - if the query asks to calculate BMI, dosage, heart rate, or perform any math.
        
        Query: {query}
        
        Output ONLY the word 'search' or 'calculate' in lowercase. Nothing else.
        """)
        
        chain = route_prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"query": state["query"]}).strip().lower()
        
        # Fallback safeguard
        if "calculate" in intent:
            intent = "calculate"
        else:
            intent = "search"
            
        print(f"[AGENT] Decision: Routing to [{intent.upper()}] tool.")
        return {"intent": intent}

    # --- NODE 2: FAISS SEARCH TOOL ---
    def _search_tool(self, state: AgentState) -> AgentState:
        print("[AGENT] Executing DP-RAG Search...")
        docs = self.retriever.invoke(state["query"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    # --- NODE 3: CLINICAL CALCULATOR TOOL ---
    def _calculator_tool(self, state: AgentState) -> AgentState:
        print("[AGENT] Executing Clinical Math Logic...")
        query = state["query"].lower()
        
        # A simple Python calculator intercepting the query
        math_result = "Could not compute. Please provide valid numbers."
        try:
            if "bmi" in query:
                # Extract numbers using basic regex for demo purposes
                numbers = re.findall(r'\d+\.?\d*', query)
                if len(numbers) >= 2:
                    weight = float(numbers[0]) # Assuming kg
                    height = float(numbers[1]) # Assuming meters
                    bmi = weight / (height ** 2)
                    math_result = f"Calculated BMI: {bmi:.1f}"
        except Exception as e:
            math_result = f"Math error: {str(e)}"
            
        return {"math_result": math_result}

    # --- NODE 4: RESPONSE GENERATOR ---
    def _generate_response(self, state: AgentState) -> AgentState:
        print("[AGENT] Synthesizing Final Answer...")
        
        gen_prompt = PromptTemplate.from_template("""
        You are a highly secure Clinical AI Assistant. Answer the user's medical query based ONLY on the provided system data below.
        If the data says 'Insufficient' or 'Could not compute', state that clearly.
        
        SYSTEM DATA:
        {data}
        
        USER QUERY:
        {query}
        
        CLINICAL ANSWER:
        """)
        
        # Determine which data to feed the LLM based on the route
        data_to_use = state.get("context") if state.get("intent") == "search" else state.get("math_result")
        
        chain = gen_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"data": data_to_use, "query": state["query"]})
        
        return {"final_answer": answer}

    # --- BUILD THE GRAPH ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add the nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("search", self._search_tool)
        workflow.add_node("calculate", self._calculator_tool)
        workflow.add_node("generator", self._generate_response)
        
        # Set the entry point
        workflow.set_entry_point("router")
        
        # Add Conditional Edges based on the Intent
        workflow.add_conditional_edges(
            "router",
            lambda state: state["intent"], # The router outputs 'search' or 'calculate'
            {
                "search": "search",
                "calculate": "calculate"
            }
        )
        
        # Both tools flow into the final generator
        workflow.add_edge("search", "generator")
        workflow.add_edge("calculate", "generator")
        workflow.add_edge("generator", END)
        
        return workflow.compile()

    # --- FASTAPI ENTRY POINT ---
    # --- FASTAPI ENTRY POINT ---
    def ask(self, query: str) -> dict:
        initial_state = {"query": query}
        result = self.app.invoke(initial_state)
        
        # Determine the source name based on the route taken
        context_text = result.get("context")
        if context_text:
            source_name = "FAISS Vector Database (Patient Records)"
        else:
            context_text = result.get("math_result", "No context generated.")
            source_name = "Python Clinical Calculator"
            
        # Wrap the result in a proper LangChain Document object to prevent FastAPI crashes!
        mock_doc = Document(
            page_content=context_text, 
            metadata={"source": source_name}
        )
        
        # Return the exact dictionary format your API expects
        return {
            "result": result["final_answer"],
            "source_documents": [mock_doc]
        }