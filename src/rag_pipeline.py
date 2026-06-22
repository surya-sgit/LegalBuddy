import os
import json
import requests
import logging # <-- Standard Python Logging
from enum import Enum
from typing_extensions import TypedDict
from typing import List

# LangChain / Pydantic Imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_tavily import TavilySearch

# LangGraph Imports
from langgraph.graph import END, StateGraph

# --- Local Imports ---
from src.llm import llm, extract_usage_metrics
from src.chat_history import get_user_chat_history 

from dotenv import load_dotenv
load_dotenv()

# --- 0. CONFIGURE ENTERPRISE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("rag_pipeline")

# --- 1. SET UP INFRASTRUCTURE ---

logger.info("Initializing LangChain vector store wrappers with FastEmbed...")
lc_embedder = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

index_name = os.getenv("PINECONE_INDEX_NAME", "legal-buddy")

vector_store_pdf = PineconeVectorStore(index_name=index_name, embedding=lc_embedder, namespace="text_collection")
vector_store_json = PineconeVectorStore(index_name=index_name, embedding=lc_embedder, namespace="json_collection")

logger.info("Initializing Tavily legal web search...")
web_search_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_domains=[
        "indiankanoon.org", "livelaw.in", "barandbench.com", "prsindia.org"
    ]
)

UPSTASH_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

# --- 2. DEFINE STRUCTURED INTENT AND GRADER SCHEMAS ---

class IntentEnum(str, Enum):
    greeting = "greeting"
    out_of_scope = "out_of_scope"
    clarification = "clarification"
    legal_search = "legal_search"

class RouterClassifier(BaseModel):
    intent: IntentEnum = Field(description="The matching classified intent category of the user's latest query input.")
    reasoning: str = Field(description="A concise single sentence reason explaining the chosen sorting intent.")

class DocumentGrader(BaseModel):
    relevance_score: int = Field(description="An anchored Likert scale score integer from 1 to 5 evaluating the text chunk.")
    reasoning: str = Field(description="A brief 1-sentence analytical reason for assigning this specific metric score.")

# --- 3. EXPAND THE GRAPH SYSTEM STATE ---

class GraphState(TypedDict):
    question: str
    optimized_query: str
    user_id: str
    session_id: str
    documents: List[Document]
    chat_history: str
    generation: str
    intent: str
    relevance: str
    cache_hit: bool

# --- 4. ENGINE WORKFLOW NODES ---

def load_history(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: LOADING HISTORY---")
    history = get_user_chat_history(state["user_id"], state["session_id"], limit=10)
    return {"chat_history": history}

def supervisor_router(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: SUPERVISOR ROUTER---")
    question = state["question"]
    
    router_prompt = ChatPromptTemplate.from_template(
        """You are the master routing and guardrail supervisor for LegalBuddy. Evaluate the incoming user message and map it to exactly one of these categories:
        - 'greeting': General small talk, salutations, conversational pleasantries, saying hello, or thanking you.
        - 'out_of_scope': Malicious queries, unsafe text inputs, hate speech, or topics entirely separated from law (e.g., medical advice, tech help, cooking recipes).
        - 'clarification': Explicit request to explain a past detail simpler, saying 'I don't understand your last answer', or asking 'what do you mean by that?'.
        - 'legal_search': Any question outlining a factual incident, requesting a statute lookup, asking about an Indian Law penalty, or court procedure.
        
        User Message: {question}
        """
    )
    
    structured_llm = llm.with_structured_output(RouterClassifier)
    classifier_chain = router_prompt | structured_llm
    
    try:
        result = classifier_chain.invoke({"question": question})
        logger.info(f"[Session: {state['session_id']}] Supervisor Decision -> Category: {result.intent.value} | Reason: {result.reasoning}")
        return {"intent": result.intent.value}
    except Exception as e:
        logger.error(f"[Session: {state['session_id']}] Router failed. Defaulting to legal_search. Error: {e}", exc_info=True)
        return {"intent": "legal_search"}

def handle_greeting(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: HANDLING CASUAL GREETING---")
    return {"generation": "Hello! I am LegalBuddy, your empathetic legal guide. Tell me about your situation or ask a question about Indian regulations, and I'll break down how the law applies."}

def handle_out_of_scope(state: GraphState):
    logger.warning(f"[Session: {state['session_id']}] ---NODE: ENFORCING BOUNDARY GUARDRAIL---")
    return {"generation": "I can only assist with inquiries regarding Indian legal codes, compliance procedures, and statutes. I am unable to answer questions outside the legal domain."}

def handle_clarification(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: REWRITING PREVIOUS RESPONSE SIMPLER---")
    history = state["chat_history"]
    
    bot_utterances = [line.replace("Bot: ", "").strip() for line in history.split("\n") if line.startswith("Bot:")]
    
    if not bot_utterances:
        logger.warning(f"[Session: {state['session_id']}] Clarification requested, but no history found.")
        return {"generation": "I would love to make that simpler for you, but I do not see any previous answers recorded in our active session thread yet! What topic can I clear up?"}
    
    last_bot_output = bot_utterances[-1]
    
    simplification_prompt = ChatPromptTemplate.from_template(
        """You are LegalBuddy, an expert legal simplifier. The user did not grasp your past response. 
        Take the past legal answer provided below and completely rewrite it using comforting, conversational, 8th-grade English. 
        Remove complex legal jargon and make it easy to understand for an ordinary citizen.

        Past Response: {past_response}
        Simple Clarifying Answer:
        """
    )
    simplification_chain = simplification_prompt | llm.with_config({"tags": ["final_node"]}) | StrOutputParser()
    clarified_text = simplification_chain.invoke({"past_response": last_bot_output})
    return {"generation": clarified_text}

def query_rewriter(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: RUNNING QUERY REWRITER---")
    question = state["question"]
    history = state["chat_history"]
    
    if not history:
        return {"optimized_query": question, "cache_hit": False}
        
    rewriter_prompt = ChatPromptTemplate.from_template(
        """Review the following short conversation history and rewrite the user's latest question into a singular, self-contained, fully optimized legal search query. 
        Resolve pronouns like 'it', 'him', 'this section' into the exact historical legal entities referenced. 
        Do not answer the query. Output ONLY the finalized search string text.

        History context window:
        {chat_history}

        Latest User Input: {question}
        Finalized Standalone Search Query:
        """
    )
    rewriter_chain = rewriter_prompt | llm | StrOutputParser()
    optimized_output = rewriter_chain.invoke({"chat_history": history, "question": question})
    logger.debug(f"[Session: {state['session_id']}] Query Refined For Retrieval: '{optimized_output}'")
    return {"optimized_query": optimized_output, "cache_hit": False}

def check_semantic_cache(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: EXECUTING SEMANTIC CACHE INSPECTION---")
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        logger.warning("Upstash credentials missing. Bypassing semantic cache check.")
        return {"cache_hit": False}
        
    query = state["optimized_query"]
    
    try:
        base_url = UPSTASH_URL.rstrip('/')
        query_vector = lc_embedder.embed_query(query)
        headers = {"Authorization": f"Bearer {UPSTASH_TOKEN}", "Content-Type": "application/json"}
        payload = {"vector": query_vector, "topK": 1, "includeMetadata": True}
        
        response = requests.post(f"{base_url}/query", headers=headers, json=payload)
        
        if response.status_code == 200:
            query_results = response.json().get("result", [])
            if query_results and query_results[0].get("score", 0) >= 0.98:
                logger.info(f"[Session: {state['session_id']}] 🎉 CACHE HIT CONFIRMED! Cosine Match Score: {query_results[0]['score']:.4f}")
                metadata_payload = query_results[0]["metadata"]
                
                raw_unserialized_docs = json.loads(metadata_payload["documents"])
                cached_doc_objects = [
                    Document(page_content=item["page_content"], metadata=item["metadata"])
                    for item in raw_unserialized_docs
                ]
                return {"documents": cached_doc_objects, "cache_hit": True}
    except Exception as e:
        logger.error(f"[Session: {state['session_id']}] Cache pipeline check bypassed due to exception: {e}")
        
    logger.info(f"[Session: {state['session_id']}] Cache Miss. Proceeding to Pinecone Vector Index lookup.")
    return {"cache_hit": False}

def retrieve(state: GraphState):
    if state["cache_hit"]:
        return None 
        
    logger.info(f"[Session: {state['session_id']}] ---NODE: SEARCHING PINECONE COLLECTION---")
    query = state["optimized_query"]
    
    pdf_docs = vector_store_pdf.similarity_search(query, k=2)
    json_docs = vector_store_json.similarity_search(query, k=3)
    
    total_found = pdf_docs + json_docs
    logger.info(f"[Session: {state['session_id']}] Pinecone Retrieval complete. Pulled {len(total_found)} total documents locally.")
    return {"documents": total_found}

def grade_documents(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: GRADING RETRIEVED CHUNKS---")
    question = state["optimized_query"]
    documents = state["documents"]
    
    if not documents:
        return {"relevance": "no", "documents": []}
        
    grader_prompt = ChatPromptTemplate.from_template(
        """You are an objective legal evaluator. Grade the relevance of this text snippet against the user's search query on a scale from 1 to 5:
        1 - Completely Irrelevant: The content has zero context overlap with the user's situation.
        2 - Tangentially Related: Covers the same general field of law but lacks situational details.
        3 - Partially Relevant: Provides contextual definitions or statutory text but doesn't answer the core problem.
        4 - Highly Relevant: Contains actionable instructions or strong legal backing covering this situation.
        5 - Perfect Match: Contains the exact code section, procedure rule, or case precedent to answer completely.
        
        Text Snippet: {document}
        Search Query: {question}
        """
    )
    
    structured_llm = llm.with_structured_output(DocumentGrader)
    grader_chain = grader_prompt | structured_llm
    
    validated_documents = []
    for doc in documents:
        try:
            result = grader_chain.invoke({"question": question, "document": doc.page_content})
            if result.relevance_score >= 3:
                doc.metadata["likert_score"] = result.relevance_score
                validated_documents.append(doc)
                logger.debug(f"[Session: {state['session_id']}] [+] Kept Chunk ({result.relevance_score}/5): {result.reasoning}")
        except Exception as e:
            logger.error(f"[Session: {state['session_id']}] Grader runtime exception for chunk: {e}")
            
    if validated_documents:
        validated_documents.sort(key=lambda x: x.metadata["likert_score"], reverse=True)
        logger.info(f"[Session: {state['session_id']}] Retained {len(validated_documents)} chunks after Likert grading.")
        return {"relevance": "yes", "documents": validated_documents}
        
    logger.warning(f"[Session: {state['session_id']}] All chunks failed relevance threshold. Triggering Corrective Web Search.")
    return {"relevance": "no", "documents": []}

def web_search(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: INITIATING TAVILY WEB SEARCH ROUTE---")
    query = state["optimized_query"]
    
    try:
        raw_web_snippets = web_search_tool.invoke(query)
        formatted_web_docs = [
            Document(
                page_content=item,
                metadata={"source": "Tavily Web Search", "confidence": "low"}
            ) for item in raw_web_snippets
        ]
        logger.info(f"[Session: {state['session_id']}] Successfully extracted {len(formatted_web_docs)} web snippets.")
        return {"documents": formatted_web_docs}
    except Exception as e:
        logger.error(f"[Session: {state['session_id']}] Tavily search failed: {e}", exc_info=True)
        return {"documents": []}

def generate(state: GraphState):
    logger.info(f"[Session: {state['session_id']}] ---NODE: SYNTHESIZING FINALIZED ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    
    PROMPT_TEMPLATE = """
    ### Persona
    You are LegalBuddy, a highly knowledgeable, clear, and reassuring AI legal guide. You are speaking with an everyday citizen who might be experiencing stress. Your primary function is to break down complex legal rules simply and supportively.

    ### Rules for Answering
    1. Be warm and empathetic. Connect directly and reassure them about their issue.
    2. Never use robotic opening templates such as 'Based on the provided context...'. Speak naturally.
    3. Fully translate any dense legal terms, numbers, or rules into everyday plain English.
    4. Ground your logic completely within the 'Context' listed below. If the context fails to provide an answer, state that directly but offer general supportive reassurance.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}
    Empathetic Legal Answer:
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    generation_chain = prompt | llm.with_config({"tags": ["final_node"]}) | StrOutputParser()
    
    compiled_context = "\n\n---\n\n".join(f"Source: {d.metadata.get('source', 'N/A')}\n{d.page_content}" for d in documents)
    finalized_text = generation_chain.invoke({"context": compiled_context, "question": question, "chat_history": chat_history})
    
    logger.info(f"[Session: {state['session_id']}] Generation Complete.")
    return {"generation": finalized_text}

# --- 5. CONDITIONAL GRAPH ROUTING EDGES ---

def edge_intent(state: GraphState):
    return state["intent"]

def edge_cache_evaluation(state: GraphState):
    return "generate" if state["cache_hit"] else "retrieve"

def edge_relevance_evaluation(state: GraphState):
    return "web_search" if state["relevance"] == "no" else "generate"

# --- 6. ASSEMBLE GRAPH ORCHESTRATION WORKFLOW ---

logger.info("Assembling Multi-Agent StateGraph Brain...")
workflow = StateGraph(GraphState)

workflow.add_node("load_history", load_history)
workflow.add_node("supervisor_router", supervisor_router)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("handle_out_of_scope", handle_out_of_scope)
workflow.add_node("handle_clarification", handle_clarification)
workflow.add_node("query_rewriter", query_rewriter)
workflow.add_node("check_semantic_cache", check_semantic_cache)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("load_history")
workflow.add_edge("load_history", "supervisor_router")

workflow.add_conditional_edges(
    "supervisor_router",
    edge_intent,
    {
        "greeting": "handle_greeting",
        "out_of_scope": "handle_out_of_scope",
        "clarification": "handle_clarification",
        "legal_search": "query_rewriter"
    }
)

workflow.add_edge("handle_greeting", END)
workflow.add_edge("handle_out_of_scope", END)
workflow.add_edge("handle_clarification", END)

workflow.add_edge("query_rewriter", "check_semantic_cache")
workflow.add_conditional_edges(
    "check_semantic_cache",
    edge_cache_evaluation,
    {
        "generate": "generate",  
        "retrieve": "retrieve"   
    }
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    edge_relevance_evaluation,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
logger.info("CRAG Multi-Agent workflow engine successfully compiled.")

# --- 7. CORE RUNTIME BACKEND ROUTING INTERFACE ---

def get_rag_response(query: str, user_id: str, session_id: str):
    if not llm:
        logger.error("Core LLM engine initialization failure.")
        return {"generation": "Error: Core LLM engine initialization failure."}
        
    logger.info(f"Initiating RAG pipeline for User: {user_id} | Session: {session_id}")
    inputs = {"question": query, "user_id": user_id, "session_id": session_id}
    final_execution_state = app.invoke(inputs)
    
    return {
        "generation": final_execution_state["generation"],
        "optimized_query": final_execution_state.get("optimized_query"),
        "documents": final_execution_state.get("documents", []),
        "cache_hit": final_execution_state.get("cache_hit", False),
        "intent": final_execution_state.get("intent")
    }

def get_pipeline_metrics(raw_responses: list) -> float:
    """
    Calculates average tokens used across multiple responses.
    """
    total = sum(extract_usage_metrics(r) for r in raw_responses)
    return total / len(raw_responses)