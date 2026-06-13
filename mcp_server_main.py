# mcp_server_main.py

# --- MCP Spec Compliance: Reject null JSON-RPC IDs ---
# The mcp SDK's JSONRPCNotification uses extra="allow", which causes
# {"id": null} to be misclassified as a notification (202 Accepted).
# Per MCP 2025-11-25, null IDs must be rejected with -32600 Invalid Request.
# Changing to extra="forbid" makes validation fail for null IDs,
# returning a proper JSON-RPC error response.
from mcp.types import JSONRPCNotification as _McpJSONRPCNotification, JSONRPCMessage as _McpJSONRPCMessage
from pydantic import ConfigDict as _ConfigDict
_McpJSONRPCNotification.model_config = _ConfigDict(extra="forbid")
_McpJSONRPCNotification.model_rebuild(force=True)
_McpJSONRPCMessage.model_rebuild(force=True)
# --- End MCP Spec Compliance ---

import asyncio
import atexit
import logging
import httpx
import json
import os
import time
from collections import defaultdict
from functools import partial
from pydantic import HttpUrl, Field
from typing import Optional, Dict, List, Literal, Any
from fastmcp.server.middleware import Middleware, MiddlewareContext

# Optional tiktoken import for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None
from fastmcp import Context

# Use standard exception for tool errors
class ToolError(Exception):
    """Tool execution error"""
    pass

# --- Logging Configuration Start ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
# --- Logging Configuration End ---

# --- Token Counting Middleware ---
class TokenCountingMiddleware(Middleware):
    """Middleware for counting input/output tokens using tiktoken."""
    
    def __init__(self, model: str = "cl100k_base"):
        """Initialize token counting middleware.

        Args:
            model: Tiktoken model name (cl100k_base for GPT-4/Claude compatibility)
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is required for token counting. Install with: pip install tiktoken")

        self.encoder = tiktoken.get_encoding(model)
        self.model = model
        self.token_stats = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})
        self.logger = logging.getLogger("token_counter")
        self.logger.setLevel(logging.INFO)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not text:
            return 0
        try:
            return len(self.encoder.encode(str(text)))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return 0
    
    def extract_text_content(self, data: Any) -> str:
        """Extract text content from various data types."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Extract text from common response fields
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict) and 'text' in item:
                            text_parts.append(str(item['text']))
            return ' '.join(text_parts)
        elif isinstance(data, list):
            text_parts = []
            for item in data:
                text_parts.append(self.extract_text_content(item))
            return ' '.join(text_parts)
        else:
            return str(data)
    
    def log_token_usage(self, operation: str, input_tokens: int, output_tokens: int, 
                       tool_name: str = None, duration_ms: float = None):
        """Log token usage with structured format."""
        log_data = {
            "operation": operation,
            "tool_name": tool_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_ms": duration_ms,
            "timestamp": time.time()
        }
        
        # Update statistics
        key = tool_name if tool_name else operation
        self.token_stats[key]["input"] += input_tokens
        self.token_stats[key]["output"] += output_tokens
        self.token_stats[key]["calls"] += 1
        
        # Log as JSON for easy parsing
        self.logger.info(json.dumps(log_data))
        
        # Also log human-readable format to main logger
        logger.info(f"Token Usage - {operation}" + 
                   (f" ({tool_name})" if tool_name else "") +
                   f": {input_tokens} in + {output_tokens} out = {input_tokens + output_tokens} total")
    
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Count tokens for tool calls."""
        start_time = time.perf_counter()
        
        # Extract tool name and arguments
        tool_name = getattr(context.message, 'name', 'unknown_tool')
        tool_args = getattr(context.message, 'arguments', {})
        
        # Count input tokens (tool arguments)
        input_text = self.extract_text_content(tool_args)
        input_tokens = self.count_tokens(input_text)
        
        try:
            # Execute the tool
            result = await call_next(context)
            
            # Count output tokens (tool result)
            output_text = self.extract_text_content(result)
            output_tokens = self.count_tokens(output_text)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log token usage
            self.log_token_usage("tool_call", input_tokens, output_tokens, 
                               tool_name, duration_ms)
            
            return result
            
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_token_usage("tool_call_error", input_tokens, 0, 
                               tool_name, duration_ms)
            raise
    
    async def on_read_resource(self, context: MiddlewareContext, call_next):
        """Count tokens for resource reads."""
        start_time = time.perf_counter()
        
        # Extract resource URI
        resource_uri = getattr(context.message, 'uri', 'unknown_resource')
        
        try:
            # Execute the resource read
            result = await call_next(context)
            
            # Count output tokens (resource content)
            output_text = self.extract_text_content(result)
            output_tokens = self.count_tokens(output_text)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log token usage (no input tokens for resource reads)
            self.log_token_usage("resource_read", 0, output_tokens, 
                               resource_uri, duration_ms)
            
            return result
            
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_token_usage("resource_read_error", 0, 0, 
                               resource_uri, duration_ms)
            raise
    
    async def on_get_prompt(self, context: MiddlewareContext, call_next):
        """Count tokens for prompt retrievals."""
        start_time = time.perf_counter()
        
        # Extract prompt name
        prompt_name = getattr(context.message, 'name', 'unknown_prompt')
        
        try:
            # Execute the prompt retrieval
            result = await call_next(context)
            
            # Count output tokens (prompt content)
            output_text = self.extract_text_content(result)
            output_tokens = self.count_tokens(output_text)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log token usage
            self.log_token_usage("prompt_get", 0, output_tokens, 
                               prompt_name, duration_ms)
            
            return result
            
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.log_token_usage("prompt_get_error", 0, 0, 
                               prompt_name, duration_ms)
            raise
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics."""
        return dict(self.token_stats)
    
    def reset_token_stats(self):
        """Reset token usage statistics."""
        self.token_stats.clear()

# --- End Token Counting Middleware ---

# Create FastMCP app directly without authentication wrapper
from fastmcp import FastMCP

def create_app():
    """Create FastMCP app with standard capabilities."""
    global app
    logger.info("MCP server created with standard capabilities...")
    
    # Add token counting middleware only if tiktoken is available
    if TIKTOKEN_AVAILABLE:
        try:
            token_counter = TokenCountingMiddleware()
            app.add_middleware(token_counter)
            logger.info("Token counting middleware added to MCP server")
        except Exception as e:
            logger.warning(f"Failed to add token counting middleware: {e}")
    
    return app

# --- Module Imports ---
from yargitay_mcp_module.client import YargitayOfficialApiClient
from bedesten_mcp_module.client import BedestenApiClient, BedestenRateLimited
from bedesten_mcp_module.models import (
    BedestenSearchRequest, BedestenSearchData,
    BedestenDocumentMarkdown, BedestenCourtTypeEnum
)
from bedesten_mcp_module.enums import BirimAdiEnum

# Semantic Search Module Imports (enabled if any embedding provider is configured)
from semantic_search.embedder import is_semantic_search_available, is_local_embedding_configured
SEMANTIC_SEARCH_AVAILABLE = is_semantic_search_available()

if SEMANTIC_SEARCH_AVAILABLE:
    from semantic_search.embedder import get_embedder
    from semantic_search.vector_store import VectorStore
    from semantic_search.processor import DocumentProcessor
    provider = "local" if is_local_embedding_configured() else "openrouter"
    logger.info(f"Semantic search enabled (provider={provider})")
else:
    logger.info("Semantic search disabled (no embedding provider configured)")

from danistay_mcp_module.client import DanistayApiClient
from emsal_mcp_module.client import EmsalApiClient
from emsal_mcp_module.models import (
    EmsalSearchRequest, CompactEmsalSearchResult
)
from uyusmazlik_mcp_module.client import UyusmazlikApiClient
from uyusmazlik_mcp_module.models import (
    UyusmazlikSearchRequest, UyusmazlikBolumEnum, UyusmazlikTuruEnum, UyusmazlikKararSonucuEnum
)
from anayasa_mcp_module.client import AnayasaMahkemesiApiClient
from anayasa_mcp_module.bireysel_client import AnayasaBireyselBasvuruApiClient
from anayasa_mcp_module.unified_client import AnayasaUnifiedClient
from anayasa_mcp_module.models import (
    AnayasaUnifiedSearchRequest,
    # Removed enum imports - now using Literal strings in models
)
# KIK v2 Module Imports (New API)
from kik_mcp_module.client_v2 import KikV2ApiClient
from kik_mcp_module.models_v2 import KikV2DecisionType

from rekabet_mcp_module.client import RekabetKurumuApiClient
from rekabet_mcp_module.models import (
    RekabetKurumuSearchRequest,
    RekabetSearchResult,
    RekabetKararTuruGuidEnum
)

from sayistay_mcp_module.client import SayistayApiClient
from sayistay_mcp_module.models import (
    SayistayUnifiedSearchRequest
)
from sayistay_mcp_module.unified_client import SayistayUnifiedClient

# KVKK Module Imports
from kvkk_mcp_module.client import KvkkApiClient
from kvkk_mcp_module.models import (
    KvkkSearchRequest,
    KvkkSearchResult,
    KvkkDocumentMarkdown
)

# BDDK Module Imports
from bddk_mcp_module.client import BddkApiClient
from bddk_mcp_module.models import (
    BddkSearchRequest
)

# GİB Module Imports
from gib_mcp_module.client import GibApiClient
from gib_mcp_module.models import (
    GibSearchRequest,
    GibSearchResult,
    GibDocumentMarkdown
)

# Sigorta Tahkim Module Imports
from sigorta_tahkim_mcp_module.client import SigortaTahkimApiClient
from sigorta_tahkim_mcp_module.models import (
    SigortaTahkimSearchRequest
)


# Create a placeholder app that will be properly initialized after tools are defined

# MCP app for Turkish legal databases with explicit capabilities
app = FastMCP(
    name="Yargı MCP Server",
    version="0.1.6"
)

# --- Health Check Functions (using individual clients) ---

# --- API Client Instances ---
yargitay_client_instance = YargitayOfficialApiClient()
danistay_client_instance = DanistayApiClient()
emsal_client_instance = EmsalApiClient()
uyusmazlik_client_instance = UyusmazlikApiClient()
anayasa_norm_client_instance = AnayasaMahkemesiApiClient()
anayasa_bireysel_client_instance = AnayasaBireyselBasvuruApiClient()
anayasa_unified_client_instance = AnayasaUnifiedClient()
kik_v2_client_instance = KikV2ApiClient()
rekabet_client_instance = RekabetKurumuApiClient()
bedesten_client_instance = BedestenApiClient()
sayistay_client_instance = SayistayApiClient()
sayistay_unified_client_instance = SayistayUnifiedClient()
kvkk_client_instance = KvkkApiClient()
bddk_client_instance = BddkApiClient()
gib_client_instance = GibApiClient()
sigorta_tahkim_client_instance = SigortaTahkimApiClient()

# Health check client (singleton for reuse)
_health_check_client: Optional[httpx.AsyncClient] = None


KARAR_TURU_ADI_TO_GUID_ENUM_MAP = {
    "": RekabetKararTuruGuidEnum.TUMU,  # Keep for backward compatibility
    "ALL": RekabetKararTuruGuidEnum.TUMU,  # Map "ALL" to TUMU
    "Birleşme ve Devralma": RekabetKararTuruGuidEnum.BIRLESME_DEVRALMA,
    "Diğer": RekabetKararTuruGuidEnum.DIGER,
    "Menfi Tespit ve Muafiyet": RekabetKararTuruGuidEnum.MENFI_TESPIT_MUAFIYET,
    "Özelleştirme": RekabetKararTuruGuidEnum.OZELLESTIRME,
    "Rekabet İhlali": RekabetKararTuruGuidEnum.REKABET_IHLALI,
}

# --- MCP Tools for Yargitay ---
"""
@app.tool(
    description="Use this when searching Turkish Court of Cassation (Yargıtay) decisions. Supports 52 chamber filtering and advanced operators (+required, -excluded, \"exact phrase\").",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_yargitay_detailed(
    arananKelime: str = Field("", description="Turkish search keyword. Supports +required -excluded \"exact phrase\" operators"),
    birimYrgKurulDaire: str = Field("ALL", description="Chamber selection (52 options: Civil/Criminal chambers, General Assemblies)"),
    esasYil: str = Field("", description="Case year for 'Esas No'."),
    esasIlkSiraNo: str = Field("", description="Starting sequence number for 'Esas No'."),
    esasSonSiraNo: str = Field("", description="Ending sequence number for 'Esas No'."),
    kararYil: str = Field("", description="Decision year for 'Karar No'."),
    kararIlkSiraNo: str = Field("", description="Starting sequence number for 'Karar No'."),
    kararSonSiraNo: str = Field("", description="Ending sequence number for 'Karar No'."),
    baslangicTarihi: str = Field("", description="Start date for decision search (DD.MM.YYYY)."),
    bitisTarihi: str = Field("", description="End date for decision search (DD.MM.YYYY)."),
    # pageSize: int = Field(10, ge=1, le=10, description="Number of results per page."),
    pageNumber: int = Field(1, ge=1, description="Page number to retrieve.")
) -> CompactYargitaySearchResult:
    # Search Yargıtay decisions using primary API with 52 chamber filtering and advanced operators.
    
    # Convert "ALL" to empty string for API compatibility
    if birimYrgKurulDaire == "ALL":
        birimYrgKurulDaire = ""
    
    pageSize = 10  # Default value
    
    search_query = YargitayDetailedSearchRequest(
        arananKelime=arananKelime,
        birimYrgKurulDaire=birimYrgKurulDaire,
        esasYil=esasYil,
        esasIlkSiraNo=esasIlkSiraNo,
        esasSonSiraNo=esasSonSiraNo,
        kararYil=kararYil,
        kararIlkSiraNo=kararIlkSiraNo,
        kararSonSiraNo=kararSonSiraNo,
        baslangicTarihi=baslangicTarihi,
        bitisTarihi=bitisTarihi,
        siralama="3",
        siralamaDirection="desc",
        pageSize=pageSize,
        pageNumber=pageNumber
    )
    
    logger.info(f"Tool 'search_yargitay_detailed' called: {search_query.model_dump_json(exclude_none=True, indent=2)}")
    try:
        api_response = await yargitay_client_instance.search_detailed_decisions(search_query)
        if api_response and api_response.data and api_response.data.data:
            # Convert to clean decision entries without arananKelime field
            clean_decisions = [
                CleanYargitayDecisionEntry(
                    id=decision.id,
                    daire=decision.daire,
                    esasNo=decision.esasNo,
                    kararNo=decision.kararNo,
                    kararTarihi=decision.kararTarihi,
                    document_url=decision.document_url
                )
                for decision in api_response.data.data
            ]
            return CompactYargitaySearchResult(
                decisions=clean_decisions,
                total_records=api_response.data.recordsTotal if api_response.data else 0,
                requested_page=search_query.pageNumber,
                page_size=search_query.pageSize)
        logger.warning("API response for Yargitay search did not contain expected data structure.")
        return CompactYargitaySearchResult(decisions=[], total_records=0, requested_page=search_query.pageNumber, page_size=search_query.pageSize)
    except Exception as e:
        logger.exception(f"Error in tool 'search_yargitay_detailed'.")
        raise

@app.tool(
    description="Use this when retrieving full text of a Yargıtay (Court of Cassation) decision. Returns clean Markdown format.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_yargitay_document_markdown(id: str) -> YargitayDocumentMarkdown:
    # Get Yargıtay decision text as Markdown. Use ID from search results.
    logger.info(f"Tool 'get_yargitay_document_markdown' called for ID: {id}")
    if not id or not id.strip(): raise ValueError("Document ID must be a non-empty string.")
    try:
        return await yargitay_client_instance.get_decision_document_as_markdown(id)
    except Exception as e:
        logger.exception(f"Error in tool 'get_yargitay_document_markdown'.")
        raise
"""

# --- MCP Tools for Danistay ---
"""
@app.tool(
    description="Use this when searching Turkish Council of State (Danıştay) decisions using AND/OR/NOT keyword logic.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_danistay_by_keyword(
    andKelimeler: List[str] = Field(default_factory=list, description="Keywords for AND logic, e.g., ['word1', 'word2']"),
    orKelimeler: List[str] = Field(default_factory=list, description="Keywords for OR logic."),
    notAndKelimeler: List[str] = Field(default_factory=list, description="Keywords for NOT AND logic."),
    notOrKelimeler: List[str] = Field(default_factory=list, description="Keywords for NOT OR logic."),
    pageNumber: int = Field(1, ge=1, description="Page number."),
    # pageSize: int = Field(10, ge=1, le=10, description="Results per page.")
) -> CompactDanistaySearchResult:
    # Search Danıştay decisions with keyword logic.
    
    pageSize = 10  # Default value
    
    search_query = DanistayKeywordSearchRequest(
        andKelimeler=andKelimeler,
        orKelimeler=orKelimeler,
        notAndKelimeler=notAndKelimeler,
        notOrKelimeler=notOrKelimeler,
        pageNumber=pageNumber,
        pageSize=pageSize
    )
    
    logger.info(f"Tool 'search_danistay_by_keyword' called.")
    try:
        api_response = await danistay_client_instance.search_keyword_decisions(search_query)
        if api_response.data:
            return CompactDanistaySearchResult(
                decisions=api_response.data.data,
                total_records=api_response.data.recordsTotal,
                requested_page=search_query.pageNumber,
                page_size=search_query.pageSize)
        logger.warning("API response for Danistay keyword search did not contain expected data structure.")
        return CompactDanistaySearchResult(decisions=[], total_records=0, requested_page=search_query.pageNumber, page_size=search_query.pageSize)
    except Exception as e:
        logger.exception(f"Error in tool 'search_danistay_by_keyword'.")
        raise

@app.tool(
    description="Use this when searching Danıştay decisions with specific chamber, case numbers, and date filters.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_danistay_detailed(
    daire: str = Field("", description="Chamber/Department name (e.g., '1. Daire')."),
    esasYil: str = Field("", description="Case year for 'Esas No'."),
    esasIlkSiraNo: str = Field("", description="Starting sequence for 'Esas No'."),
    esasSonSiraNo: str = Field("", description="Ending sequence for 'Esas No'."),
    kararYil: str = Field("", description="Decision year for 'Karar No'."),
    kararIlkSiraNo: str = Field("", description="Starting sequence for 'Karar No'."),
    kararSonSiraNo: str = Field("", description="Ending sequence for 'Karar No'."),
    baslangicTarihi: str = Field("", description="Start date for decision (DD.MM.YYYY)."),
    bitisTarihi: str = Field("", description="End date for decision (DD.MM.YYYY)."),
    mevzuatNumarasi: str = Field("", description="Legislation number."),
    mevzuatAdi: str = Field("", description="Legislation name."),
    madde: str = Field("", description="Article number."),
    pageNumber: int = Field(1, ge=1, description="Page number."),
    # pageSize: int = Field(10, ge=1, le=10, description="Results per page.")
) -> CompactDanistaySearchResult:
    # Search Danıştay decisions with detailed filtering.
    
    pageSize = 10  # Default value
    
    search_query = DanistayDetailedSearchRequest(
        daire=daire,
        esasYil=esasYil,
        esasIlkSiraNo=esasIlkSiraNo,
        esasSonSiraNo=esasSonSiraNo,
        kararYil=kararYil,
        kararIlkSiraNo=kararIlkSiraNo,
        kararSonSiraNo=kararSonSiraNo,
        baslangicTarihi=baslangicTarihi,
        bitisTarihi=bitisTarihi,
        mevzuatNumarasi=mevzuatNumarasi,
        mevzuatAdi=mevzuatAdi,
        madde=madde,
        siralama="3",
        siralamaDirection="desc",
        pageNumber=pageNumber,
        pageSize=pageSize
    )
    
    logger.info(f"Tool 'search_danistay_detailed' called.")
    try:
        api_response = await danistay_client_instance.search_detailed_decisions(search_query)
        if api_response.data:
            return CompactDanistaySearchResult(
                decisions=api_response.data.data,
                total_records=api_response.data.recordsTotal,
                requested_page=search_query.pageNumber,
                page_size=search_query.pageSize)
        logger.warning("API response for Danistay detailed search did not contain expected data structure.")
        return CompactDanistaySearchResult(decisions=[], total_records=0, requested_page=search_query.pageNumber, page_size=search_query.pageSize)
    except Exception as e:
        logger.exception(f"Error in tool 'search_danistay_detailed'.")
        raise

@app.tool(
    description="Use this when retrieving full text of a Danıştay (Council of State) decision. Returns clean Markdown format.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_danistay_document_markdown(id: str) -> DanistayDocumentMarkdown:
    # Get Danıştay decision text as Markdown. Use ID from search results.
    logger.info(f"Tool 'get_danistay_document_markdown' called for ID: {id}")
    if not id or not id.strip(): raise ValueError("Document ID must be a non-empty string for Danıştay.")
    try:
        return await danistay_client_instance.get_decision_document_as_markdown(id)
    except Exception as e:
        logger.exception(f"Error in tool 'get_danistay_document_markdown'.")
        raise
"""

# --- MCP Tools for Emsal ---
@app.tool(
    description="Use this when searching UYAP precedent decisions (Emsal). For lower court decisions and case law.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_emsal_detailed_decisions(
    keyword: str = Field("", description="Keyword to search."),
    selected_bam_civil_court: str = Field("", description="Selected BAM Civil Court."),
    selected_civil_court: str = Field("", description="Selected Civil Court."),
    selected_regional_civil_chambers: List[str] = Field(default_factory=list, description="Selected Regional Civil Chambers."),
    case_year_esas: str = Field("", description="Case year for 'Esas No'."),
    case_start_seq_esas: str = Field("", description="Starting sequence for 'Esas No'."),
    case_end_seq_esas: str = Field("", description="Ending sequence for 'Esas No'."),
    decision_year_karar: str = Field("", description="Decision year for 'Karar No'."),
    decision_start_seq_karar: str = Field("", description="Starting sequence for 'Karar No'."),
    decision_end_seq_karar: str = Field("", description="Ending sequence for 'Karar No'."),
    start_date: str = Field("", description="Start date for decision (DD.MM.YYYY)."),
    end_date: str = Field("", description="End date for decision (DD.MM.YYYY)."),
    sort_criteria: str = Field("1", description="Sorting criteria (e.g., 1: Esas No)."),
    sort_direction: str = Field("desc", description="Sorting direction ('asc' or 'desc')."),
    page_number: int = Field(1, ge=1, description="Page number (accepts int)."),
    # page_size: int = Field(10, ge=1, le=10, description="Results per page.")
) -> Dict[str, Any]:
    """Search Emsal precedent decisions with detailed criteria."""
    
    page_size = 10  # Default value
    
    search_query = EmsalSearchRequest(
        keyword=keyword,
        selected_bam_civil_court=selected_bam_civil_court,
        selected_civil_court=selected_civil_court,
        selected_regional_civil_chambers=selected_regional_civil_chambers,
        case_year_esas=case_year_esas,
        case_start_seq_esas=case_start_seq_esas,
        case_end_seq_esas=case_end_seq_esas,
        decision_year_karar=decision_year_karar,
        decision_start_seq_karar=decision_start_seq_karar,
        decision_end_seq_karar=decision_end_seq_karar,
        start_date=start_date,
        end_date=end_date,
        sort_criteria=sort_criteria,
        sort_direction=sort_direction,
        page_number=page_number,
        page_size=page_size
    )
    
    logger.info("Tool 'search_emsal_detailed_decisions' called.")
    try:
        api_response = await emsal_client_instance.search_detailed_decisions(search_query)
        if api_response.data:
            return CompactEmsalSearchResult(
                decisions=api_response.data.data,
                total_records=api_response.data.recordsTotal if api_response.data.recordsTotal is not None else 0,
                requested_page=search_query.page_number,
                page_size=search_query.page_size
            ).model_dump()
        logger.warning("API response for Emsal search did not contain expected data structure.")
        return CompactEmsalSearchResult(decisions=[], total_records=0, requested_page=search_query.page_number, page_size=search_query.page_size).model_dump()
    except Exception:
        logger.exception("Error in tool 'search_emsal_detailed_decisions'.")
        raise

@app.tool(
    description="Use this when retrieving full text of an Emsal precedent decision. Returns clean Markdown format.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_emsal_document_markdown(id: str) -> Dict[str, Any]:
    """Get document as Markdown."""
    logger.info(f"Tool 'get_emsal_document_markdown' called for ID: {id}")
    if not id or not id.strip(): raise ValueError("Document ID required for Emsal.")
    try:
        result = await emsal_client_instance.get_decision_document_as_markdown(id)
        return result.model_dump()
    except Exception:
        logger.exception("Error in tool 'get_emsal_document_markdown'.")
        raise

# --- MCP Tools for Uyusmazlik ---
@app.tool(
    description="Use this when searching jurisdictional dispute court (Uyuşmazlık Mahkemesi) decisions. Resolves conflicts between civil and administrative courts.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_uyusmazlik_decisions(
    icerik: str = Field("", description="Keyword or content for main text search."),
    bolum: Literal["ALL", "Ceza Bölümü", "Genel Kurul Kararları", "Hukuk Bölümü"] = Field("ALL", description="Select the department (Bölüm). Use 'ALL' for all departments."),
    uyusmazlik_turu: Literal["ALL", "Görev Uyuşmazlığı", "Hüküm Uyuşmazlığı"] = Field("ALL", description="Select the type of dispute. Use 'ALL' for all types."),
    karar_sonuclari: List[Literal["Hüküm Uyuşmazlığı Olmadığına Dair", "Hüküm Uyuşmazlığı Olduğuna Dair"]] = Field(default_factory=list, description="List of desired 'Karar Sonucu' types."),
    esas_yil: str = Field("", description="Case year ('Esas Yılı')."),
    esas_sayisi: str = Field("", description="Case number ('Esas Sayısı')."),
    karar_yil: str = Field("", description="Decision year ('Karar Yılı')."),
    karar_sayisi: str = Field("", description="Decision number ('Karar Sayısı')."),
    kanun_no: str = Field("", description="Relevant Law Number."),
    karar_date_begin: str = Field("", description="Decision start date (DD.MM.YYYY)."),
    karar_date_end: str = Field("", description="Decision end date (DD.MM.YYYY)."),
    resmi_gazete_sayi: str = Field("", description="Official Gazette number."),
    resmi_gazete_date: str = Field("", description="Official Gazette date (DD.MM.YYYY)."),
    tumce: str = Field("", description="Exact phrase search."),
    wild_card: str = Field("", description="Search for phrase and its inflections."),
    hepsi: str = Field("", description="Search for texts containing all specified words."),
    herhangi_birisi: str = Field("", description="Search for texts containing any of the specified words."),
    not_hepsi: str = Field("", description="Exclude texts containing these specified words.")
) -> Dict[str, Any]:
    """Search Court of Jurisdictional Disputes decisions."""
    
    # Convert string literals to enums
    # Map "ALL" to TUMU for backward compatibility
    if bolum == "ALL":
        bolum_enum = UyusmazlikBolumEnum.TUMU
    else:
        bolum_enum = UyusmazlikBolumEnum(bolum) if bolum else UyusmazlikBolumEnum.TUMU
    
    if uyusmazlik_turu == "ALL":
        uyusmazlik_turu_enum = UyusmazlikTuruEnum.TUMU
    else:
        uyusmazlik_turu_enum = UyusmazlikTuruEnum(uyusmazlik_turu) if uyusmazlik_turu else UyusmazlikTuruEnum.TUMU
    karar_sonuclari_enums = [UyusmazlikKararSonucuEnum(ks) for ks in karar_sonuclari]
    
    search_params = UyusmazlikSearchRequest(
        icerik=icerik,
        bolum=bolum_enum,
        uyusmazlik_turu=uyusmazlik_turu_enum,
        karar_sonuclari=karar_sonuclari_enums,
        esas_yil=esas_yil,
        esas_sayisi=esas_sayisi,
        karar_yil=karar_yil,
        karar_sayisi=karar_sayisi,
        kanun_no=kanun_no,
        karar_date_begin=karar_date_begin,
        karar_date_end=karar_date_end,
        resmi_gazete_sayi=resmi_gazete_sayi,
        resmi_gazete_date=resmi_gazete_date,
        tumce=tumce,
        wild_card=wild_card,
        hepsi=hepsi,
        herhangi_birisi=herhangi_birisi,
        not_hepsi=not_hepsi
    )
    
    logger.info("Tool 'search_uyusmazlik_decisions' called.")
    try:
        result = await uyusmazlik_client_instance.search_decisions(search_params)
        return result.model_dump()
    except Exception:
        logger.exception("Error in tool 'search_uyusmazlik_decisions'.")
        raise

@app.tool(
    description="Use this when retrieving full text of an Uyuşmazlık Mahkemesi decision. Returns clean Markdown format.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_uyusmazlik_document_markdown_from_url(
    document_url: str = Field(..., description="Full URL to the Uyuşmazlık Mahkemesi decision document from search results")
) -> Dict[str, Any]:
    """Get Uyuşmazlık Mahkemesi decision as Markdown."""
    logger.info(f"Tool 'get_uyusmazlik_document_markdown_from_url' called for URL: {str(document_url)}")
    if not document_url:
        raise ValueError("Document URL (document_url) is required for Uyuşmazlık document retrieval.")
    try:
        result = await uyusmazlik_client_instance.get_decision_document_as_markdown(str(document_url))
        return result.model_dump()
    except Exception:
        logger.exception("Error in tool 'get_uyusmazlik_document_markdown_from_url'.")
        raise

# --- DEACTIVATED: MCP Tools for Anayasa Mahkemesi (Individual Tools) ---
# Use search_anayasa_unified and get_anayasa_document_unified instead

"""
@app.tool(
    description="Use this when searching Turkish Constitutional Court norm control decisions. For constitutional review and legislation challenges.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
# DEACTIVATED TOOL - Use search_anayasa_unified instead
# @app.tool(
#     description="DEACTIVATED - Use search_anayasa_unified instead",
#     annotations={"readOnlyHint": True, "openWorldHint": False, "idempotentHint": True}
# )
# async def search_anayasa_norm_denetimi_decisions(...) -> AnayasaSearchResult:
#     raise ValueError("This tool is deactivated. Use search_anayasa_unified instead.")

# DEACTIVATED TOOL - Use get_anayasa_document_unified instead
# @app.tool(...)
# async def get_anayasa_norm_denetimi_document_markdown(...) -> AnayasaDocumentMarkdown:
#     raise ValueError("This tool is deactivated. Use get_anayasa_document_unified instead.")

# DEACTIVATED TOOL - Use search_anayasa_unified instead
# @app.tool(...)
# async def search_anayasa_bireysel_basvuru_report(...) -> AnayasaBireyselReportSearchResult:
#     raise ValueError("This tool is deactivated. Use search_anayasa_unified instead.")

# DEACTIVATED TOOL - Use get_anayasa_document_unified instead
# @app.tool(...)
# async def get_anayasa_bireysel_basvuru_document_markdown(...) -> AnayasaBireyselBasvuruDocumentMarkdown:
#     raise ValueError("This tool is deactivated. Use get_anayasa_document_unified instead.")
"""

# --- Unified MCP Tools for Anayasa Mahkemesi ---
@app.tool(
    description=(
        "Use this when searching Turkish Constitutional Court decision records. Supports norm control decisions "
        "and individual application decisions. Norm control filters include reviewed norm metadata; results are court decisions."
    ),
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_anayasa_unified(
    decision_type: Literal["norm_denetimi", "bireysel_basvuru"] = Field(..., description="Decision type: norm_denetimi (norm control) or bireysel_basvuru (individual applications)"),
    keywords: List[str] = Field(default_factory=list, description="Keywords to search for (common parameter)"),
    page_to_fetch: int = Field(1, ge=1, le=100, description="Page number to fetch (1-100)"),
    # results_per_page: int = Field(10, ge=1, le=100, description="Results per page (1-100)"),
    
    # Norm Denetimi specific parameters (ignored for bireysel_basvuru)
    keywords_all: List[str] = Field(default_factory=list, description="All keywords must be present (norm_denetimi only)"),
    keywords_any: List[str] = Field(default_factory=list, description="Any of these keywords (norm_denetimi only)"),
    decision_type_norm: Literal["ALL", "1", "2", "3"] = Field("ALL", description="Decision type for norm denetimi"),
    application_date_start: str = Field("", description="Application start date (norm_denetimi only)"),
    application_date_end: str = Field("", description="Application end date (norm_denetimi only)"),
    
    # Bireysel Başvuru specific parameters (ignored for norm_denetimi)
    decision_start_date: str = Field("", description="Decision start date (bireysel_basvuru only)"),
    decision_end_date: str = Field("", description="Decision end date (bireysel_basvuru only)"),
    norm_type: Literal["ALL", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "0"] = Field("ALL", description="Norm type (bireysel_basvuru only)"),
    subject_category: str = Field("", description="Subject category (bireysel_basvuru only)")
) -> str:
    logger.info(f"Tool 'search_anayasa_unified' called for decision_type: {decision_type}")
    
    results_per_page = 10  # Default value
    
    try:
        request = AnayasaUnifiedSearchRequest(
            decision_type=decision_type,
            keywords=keywords,
            page_to_fetch=page_to_fetch,
            results_per_page=results_per_page,
            keywords_all=keywords_all,
            keywords_any=keywords_any,
            decision_type_norm=decision_type_norm,
            application_date_start=application_date_start,
            application_date_end=application_date_end,
            decision_start_date=decision_start_date,
            decision_end_date=decision_end_date,
            norm_type=norm_type,
            subject_category=subject_category
        )
        
        result = await anayasa_unified_client_instance.search_unified(request)
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        
    except Exception:
        logger.exception("Error in tool 'search_anayasa_unified'.")
        raise

@app.tool(
    description="Use this when retrieving full text of a Constitutional Court decision. Auto-detects decision type from URL.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def get_anayasa_document_unified(
    document_url: str = Field(..., description="Document URL from search results"),
    page_number: int = Field(1, ge=1, description="Page number for paginated content (1-indexed)")
) -> str:
    logger.info(f"Tool 'get_anayasa_document_unified' called for URL: {document_url}, Page: {page_number}")
    
    try:
        result = await anayasa_unified_client_instance.get_document_unified(document_url, page_number)
        return json.dumps(result.model_dump(mode='json'), ensure_ascii=False, indent=2)
        
    except Exception:
        logger.exception("Error in tool 'get_anayasa_document_unified'.")
        raise

# --- MCP Tools for KIK v2 (Kamu İhale Kurulu - New API) ---
@app.tool(
    description="Use this when searching Turkish public procurement disputes (KİK). Supports dispute, regulatory, and court decision types.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_kik_v2_decisions(
    decision_type: str = Field("uyusmazlik", description="Decision type: 'uyusmazlik' (disputes), 'duzenleyici' (regulatory), or 'mahkeme' (court decisions)"),
    karar_metni: str = Field("", description="Decision text search query"),
    karar_no: str = Field("", description="Decision number (e.g., '2025/UH.II-1801')"),
    basvuran: str = Field("", description="Applicant name"),
    idare_adi: str = Field("", description="Administration/procuring entity name"),
    baslangic_tarihi: str = Field("", description="Start date (YYYY-MM-DD format, e.g., '2025-01-01')"),
    bitis_tarihi: str = Field("", description="End date (YYYY-MM-DD format, e.g., '2025-12-31')")
) -> dict:
    """Search Public Procurement Authority (KİK) decisions using the new v2 API.
    
    This tool supports all three KİK decision types:
    - uyusmazlik: Disputes and conflicts in public procurement
    - duzenleyici: Regulatory decisions and guidelines  
    - mahkeme: Court decisions and legal interpretations
    
    Each decision type uses its respective endpoint (GetKurulKararlari, GetKurulKararlariDk, GetKurulKararlariMk)
    and returns results with the decision_type field populated for identification.
    """
    
    logger.info(f"Tool 'search_kik_v2_decisions' called with decision_type='{decision_type}', karar_metni='{karar_metni}', karar_no='{karar_no}'")
    
    try:
        # Validate and convert decision type
        try:
            kik_decision_type = KikV2DecisionType(decision_type)
        except ValueError:
            return {
                "decisions": [],
                "total_records": 0,
                "page": 1,
                "error_code": "INVALID_DECISION_TYPE",
                "error_message": f"Invalid decision type: {decision_type}. Valid options: uyusmazlik, duzenleyici, mahkeme"
            }
        
        api_response = await kik_v2_client_instance.search_decisions(
            decision_type=kik_decision_type,
            karar_metni=karar_metni,
            karar_no=karar_no,
            basvuran=basvuran,
            idare_adi=idare_adi,
            baslangic_tarihi=baslangic_tarihi,
            bitis_tarihi=bitis_tarihi
        )
        
        # Convert to dictionary for MCP tool response
        result = {
            "decisions": [decision.model_dump() for decision in api_response.decisions],
            "total_records": api_response.total_records,
            "page": api_response.page,
            "error_code": api_response.error_code,
            "error_message": api_response.error_message
        }
        
        logger.info(f"KİK v2 {decision_type} search completed. Found {len(api_response.decisions)} decisions")
        return result
        
    except Exception as e:
        logger.exception(f"Error in KİK v2 {decision_type} search tool 'search_kik_v2_decisions'.")
        return {
            "decisions": [],
            "total_records": 0,
            "page": 1,
            "error_code": "TOOL_ERROR",
            "error_message": str(e)
        }

@app.tool(
    description="Use this when retrieving full text of a KİK procurement decision. Returns document in Markdown format.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_kik_v2_document_markdown(
    gundemMaddesiId: str = Field(..., description="gundemMaddesiId from search_kik_v2_decisions results")
) -> dict:
    """Get KİK decision document in Markdown format."""

    logger.info(f"Tool 'get_kik_v2_document_markdown' called for gundemMaddesiId: {gundemMaddesiId}")

    if not gundemMaddesiId or not gundemMaddesiId.strip():
        return {
            "document_id": gundemMaddesiId,
            "kararNo": "",
            "markdown_content": "",
            "source_url": "",
            "error_message": "gundemMaddesiId is required and must be a non-empty string"
        }

    try:
        api_response = await kik_v2_client_instance.get_document_markdown(gundemMaddesiId)

        return {
            "document_id": api_response.document_id,
            "kararNo": api_response.kararNo,
            "markdown_content": api_response.markdown_content,
            "source_url": api_response.source_url,
            "error_message": api_response.error_message
        }

    except Exception as e:
        logger.exception(f"Error in KİK v2 document retrieval tool for gundemMaddesiId: {gundemMaddesiId}")
        return {
            "document_id": gundemMaddesiId,
            "kararNo": "",
            "markdown_content": "",
            "source_url": "",
            "error_message": f"Tool-level error during document retrieval: {str(e)}"
        }
@app.tool(
    description="Use this when searching Turkish competition law and antitrust decisions (Rekabet Kurumu).",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_rekabet_kurumu_decisions(
    sayfaAdi: str = Field("", description="Search in decision title (Başlık)."),
    YayinlanmaTarihi: str = Field("", description="Publication date (Yayım Tarihi), e.g., DD.MM.YYYY."),
    PdfText: str = Field(
        "",
        description='Search in decision text. Use "\\"kesin cümle\\"" for precise matching.'
    ),
    KararTuru: Literal[ 
        "ALL", 
        "Birleşme ve Devralma",
        "Diğer",
        "Menfi Tespit ve Muafiyet",
        "Özelleştirme",
        "Rekabet İhlali"
    ] = Field("ALL", description="Parameter description"),
    KararSayisi: str = Field("", description="Decision number (Karar Sayısı)."),
    KararTarihi: str = Field("", description="Decision date (Karar Tarihi), e.g., DD.MM.YYYY."),
    page: int = Field(1, ge=1, description="Page number to fetch for the results list.")
) -> Dict[str, Any]:
    """Search Competition Authority decisions."""
    
    karar_turu_guid_enum = KARAR_TURU_ADI_TO_GUID_ENUM_MAP.get(KararTuru)

    try:
        if karar_turu_guid_enum is None: 
            logger.warning(f"Invalid user-provided KararTuru: '{KararTuru}'. Defaulting to TUMU (all).")
            karar_turu_guid_enum = RekabetKararTuruGuidEnum.TUMU
    except Exception as e_map: 
        logger.error(f"Error mapping KararTuru '{KararTuru}': {e_map}. Defaulting to TUMU.")
        karar_turu_guid_enum = RekabetKararTuruGuidEnum.TUMU

    search_query = RekabetKurumuSearchRequest(
        sayfaAdi=sayfaAdi,
        YayinlanmaTarihi=YayinlanmaTarihi,
        PdfText=PdfText,
        KararTuruID=karar_turu_guid_enum, 
        KararSayisi=KararSayisi,
        KararTarihi=KararTarihi,
        page=page
    )
    logger.info(f"Tool 'search_rekabet_kurumu_decisions' called. Query: {search_query.model_dump_json(exclude_none=True, indent=2)}")
    try:
       
        result = await rekabet_client_instance.search_decisions(search_query)
        return result.model_dump()
    except Exception:
        logger.exception("Error in tool 'search_rekabet_kurumu_decisions'.")
        return RekabetSearchResult(decisions=[], retrieved_page_number=page, total_records_found=0, total_pages=0).model_dump()

@app.tool(
    description="Use this when retrieving full text of a Competition Authority decision. Returns paginated Markdown format.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_rekabet_kurumu_document(
    karar_id: str = Field(..., description="GUID (kararId) of the Rekabet Kurumu decision. This ID is obtained from search results."),
    page_number: int = Field(1, ge=1, description="Requested page number for the Markdown content converted from PDF (1-indexed, accepts int). Default is 1.")
) -> Dict[str, Any]:
    """Get Competition Authority decision as paginated Markdown."""
    logger.info(f"Tool 'get_rekabet_kurumu_document' called. Karar ID: {karar_id}, Markdown Page: {page_number}")
    
    current_page_to_fetch = page_number if page_number >= 1 else 1
    
    try:
        result = await rekabet_client_instance.get_decision_document(karar_id, page_number=current_page_to_fetch)
        return result.model_dump()
    except Exception:
        logger.exception(f"Error in tool 'get_rekabet_kurumu_document'. Karar ID: {karar_id}")
        raise 

# --- MCP Tools for Bedesten (Unified Search Across All Courts) ---
@app.tool(
    description=(
        "Use this for Turkish court decision records from Yargıtay, Danıştay, Local Courts, Appeals Courts, and KYB via Bedesten. "
        "Prefer narrow court_types over all courts. pageSize is intentionally fixed to 10 results per page. "
        "Bedesten is upstream rate-limited; avoid parallel repeated calls and wait retry_after seconds after 429 responses."
    ),
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_bedesten_unified(
    ctx: Context,
    phrase: str = Field(..., description="""Search query in Turkish. SUPPORTED OPERATORS:
• Simple: "mülkiyet hakkı" (finds both words)
• Exact phrase: "\"mülkiyet hakkı\"" (finds exact phrase)  
• Required term: "+mülkiyet hakkı" (must contain mülkiyet)
• Exclude term: "mülkiyet -kira" (contains mülkiyet but not kira)
• Boolean AND: "mülkiyet AND hak" (both terms required)
• Boolean OR: "mülkiyet OR tapu" (either term acceptable)
• Boolean NOT: "mülkiyet NOT satış" (contains mülkiyet but not satış)
NOTE: Wildcards (*,?), regex patterns (/regex/), fuzzy search (~), and proximity search are NOT supported.
For best results, use exact phrases with quotes for legal terms."""),
    court_types: List[BedestenCourtTypeEnum] = Field(
        default=["YARGITAYKARARI", "DANISTAYKARAR"], 
        description="Court types: YARGITAYKARARI, DANISTAYKARAR, YERELHUKUK, ISTINAFHUKUK, KYB"
    ),
    # pageSize: int = Field(10, ge=1, le=10, description="Results per page (1-10)"),
    pageNumber: int = Field(1, ge=1, description="Page number. Each page returns 10 results; pageSize is fixed by the server."),
    birimAdi: BirimAdiEnum = Field("ALL", description="""
        Chamber filter (optional). Abbreviated values with Turkish names:
        • Yargıtay: H1-H23 (1-23. Hukuk Dairesi), C1-C23 (1-23. Ceza Dairesi), HGK (Hukuk Genel Kurulu), CGK (Ceza Genel Kurulu), BGK (Büyük Genel Kurulu), HBK (Hukuk Daireleri Başkanlar Kurulu), CBK (Ceza Daireleri Başkanlar Kurulu)
        • Danıştay: D1-D17 (1-17. Daire), DBGK (Büyük Gen.Kur.), IDDK (İdare Dava Daireleri Kurulu), VDDK (Vergi Dava Daireleri Kurulu), IBK (İçtihatları Birleştirme Kurulu), IIK (İdari İşler Kurulu), DBK (Başkanlar Kurulu), AYIM (Askeri Yüksek İdare Mahkemesi), AYIM1-3 (Askeri Yüksek İdare Mahkemesi 1-3. Daire)
        """),
    kararTarihiStart: str = Field("", description="Start date (ISO 8601 format)"),
    kararTarihiEnd: str = Field("", description="End date (ISO 8601 format)")
) -> dict:
    """Search Turkish legal databases via unified Bedesten API."""
    
    pageSize = 10  # Default value
    
    # Convert date formats if provided
    # Accept formats: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.000Z
    if kararTarihiStart and not kararTarihiStart.endswith('Z'):
        # Convert simple date format to ISO 8601 with timezone
        if 'T' not in kararTarihiStart:
            kararTarihiStart = f"{kararTarihiStart}T00:00:00.000Z"
    
    if kararTarihiEnd and not kararTarihiEnd.endswith('Z'):
        # Convert simple date format to ISO 8601 with timezone
        if 'T' not in kararTarihiEnd:
            kararTarihiEnd = f"{kararTarihiEnd}T23:59:59.999Z"
    
    search_data = BedestenSearchData(
        pageSize=pageSize,
        pageNumber=pageNumber,
        itemTypeList=court_types,
        phrase=phrase,
        birimAdi=birimAdi,
        kararTarihiStart=kararTarihiStart,
        kararTarihiEnd=kararTarihiEnd
    )
    
    search_request = BedestenSearchRequest(data=search_data)
    
    logger.info(f"Searching bedesten: phrase='{phrase}', court_types={court_types}, birimAdi='{birimAdi}', page={pageNumber}")
    
    try:
        response = await bedesten_client_instance.search_documents(search_request)

        if response.data is None:
            return {
                "decisions": [],
                "total_records": 0,
                "requested_page": pageNumber,
                "page_size": pageSize,
                "searched_courts": court_types,
                "error": "No data returned from Bedesten API"
            }

        # Add null safety checks for response.data fields
        emsal_karar_list = response.data.emsalKararList if hasattr(response.data, 'emsalKararList') and response.data.emsalKararList is not None else []
        total_records = response.data.total if hasattr(response.data, 'total') and response.data.total is not None else 0

        return {
            "decisions": [d.model_dump() for d in emsal_karar_list],
            "total_records": total_records,
            "requested_page": pageNumber,
            "page_size": pageSize,
            "searched_courts": court_types
        }
    except BedestenRateLimited as e:
        retry_after = f"{e.retry_after:.1f}"
        logger.warning(f"Bedesten local rate-limit bucket full for search; retry-after={retry_after}s")
        return {
            "decisions": [],
            "total_records": 0,
            "requested_page": pageNumber,
            "page_size": pageSize,
            "searched_courts": court_types,
            "error": "rate_limit_exceeded",
            "status_code": 429,
            "retry_after": retry_after,
            "message": (
                "Bedesten istemci tarafı eşzamanlılık sınırına ulaşıldı "
                "(yerel token-bucket dolu). Lütfen kısa bir süre bekleyip "
                "aramayı tekrar deneyin. Yargı MCP'nin daha hızlı ve "
                "profesyonel versiyonunu test etmek için beta sürümüne "
                "kaydolabilirsiniz: "
                "https://yargi.betaspacestudio.com"
            ),
        }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After", "")
            logger.warning(f"Bedesten API rate limit (429) for search; retry-after={retry_after!r}")
            return {
                "decisions": [],
                "total_records": 0,
                "requested_page": pageNumber,
                "page_size": pageSize,
                "searched_courts": court_types,
                "error": "rate_limit_exceeded",
                "status_code": 429,
                "retry_after": retry_after,
                "message": (
                    "Bedesten API rate limit aşıldı (HTTP 429 Too Many Requests). "
                    "Lütfen kısa bir süre bekleyip aramayı tekrar deneyin. "
                    "Yargı MCP'nin daha hızlı ve profesyonel versiyonunu test "
                    "etmek için beta sürümüne kaydolabilirsiniz: "
                    "https://yargi.betaspacestudio.com"
                ),
            }
        logger.exception("Error in tool 'search_bedesten_unified'")
        raise
    except Exception:
        logger.exception("Error in tool 'search_bedesten_unified'")
        raise

@app.tool(
    description=(
        "Use this when retrieving full text of a Bedesten search result by documentId. "
        "Counts against the same Bedesten upstream rate limit as search; after 429, wait retry_after seconds before retrying."
    ),
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def get_bedesten_document_markdown(
    documentId: str = Field(..., description="Document ID from Bedesten search results")
) -> BedestenDocumentMarkdown:
    """Get legal decision document as Markdown from Bedesten API."""
    logger.info(f"Tool 'get_bedesten_document_markdown' called for ID: {documentId}")
    
    if not documentId or not documentId.strip():
        raise ValueError("Document ID must be a non-empty string.")
    
    try:
        return await bedesten_client_instance.get_document_as_markdown(documentId)
    except BedestenRateLimited as e:
        retry_after = f"{e.retry_after:.1f}"
        logger.warning(f"Bedesten local rate-limit bucket full for document {documentId}; retry-after={retry_after}s")
        message = (
            "Bedesten istemci tarafı eşzamanlılık sınırına ulaşıldı "
            "(yerel token-bucket dolu). Lütfen kısa bir süre bekleyip "
            "belgeyi tekrar talep edin. Yargı MCP'nin daha hızlı ve "
            "profesyonel versiyonunu test etmek için beta sürümüne "
            "kaydolabilirsiniz: "
            "https://yargi.betaspacestudio.com "
            f"Retry-After: {retry_after}"
        )
        return BedestenDocumentMarkdown(
            documentId=documentId,
            markdown_content=f"ERROR (rate_limit_exceeded, HTTP 429): {message}",
            source_url=f"https://mevzuat.adalet.gov.tr/ictihat/{documentId}",
            mime_type=None,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After", "")
            logger.warning(f"Bedesten API rate limit (429) for document {documentId}; retry-after={retry_after!r}")
            message = (
                "Bedesten API rate limit aşıldı (HTTP 429 Too Many Requests). "
                "Lütfen kısa bir süre bekleyip belgeyi tekrar talep edin. "
                "Yargı MCP'nin daha hızlı ve profesyonel versiyonunu test "
                "etmek için beta sürümüne kaydolabilirsiniz: "
                "https://yargi.betaspacestudio.com"
            )
            if retry_after:
                message += f" Retry-After: {retry_after}"
            return BedestenDocumentMarkdown(
                documentId=documentId,
                markdown_content=f"ERROR (rate_limit_exceeded, HTTP 429): {message}",
                source_url=f"https://mevzuat.adalet.gov.tr/ictihat/{documentId}",
                mime_type=None,
            )
        logger.exception("Error in tool 'get_bedesten_document_markdown'")
        raise
    except Exception:
        logger.exception("Error in tool 'get_bedesten_document_markdown'")
        raise


# --- Semantic Search Tool (Conditional - requires an embedding provider) ---
if SEMANTIC_SEARCH_AVAILABLE:
    from semantic_search.embedder import get_embedding_request_timeout_s

    def _semantic_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
        raw_value = os.getenv(name)
        if raw_value is None or raw_value == "":
            return default
        try:
            parsed = float(raw_value)
        except ValueError:
            logger.warning("%s=%r is invalid; using default %.1f", name, raw_value, default)
            return default
        if parsed < minimum:
            logger.warning("%s=%s below minimum %.1f; clamping", name, raw_value, minimum)
            return minimum
        if parsed > maximum:
            logger.warning("%s=%s above maximum %.1f; clamping", name, raw_value, maximum)
            return maximum
        return parsed

    def _semantic_now_ms(start_time: float) -> int:
        return int((time.monotonic() - start_time) * 1000)

    def _semantic_remaining_s(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _court_type_value(court_type: Any) -> str:
        return getattr(court_type, "value", str(court_type))

    def _decision_court_type(decision: Any) -> Optional[str]:
        item_type = getattr(decision, "itemType", None)
        return getattr(item_type, "name", None) if item_type else None

    def _semantic_title(metadata: Dict[str, Any], fallback_id: Optional[str] = None) -> str:
        title_parts = []
        if metadata.get("birim_adi"):
            title_parts.append(metadata["birim_adi"])
        if metadata.get("esas_no"):
            title_parts.append(f"Esas: {metadata['esas_no']}")
        if metadata.get("karar_no"):
            title_parts.append(f"Karar: {metadata['karar_no']}")
        if metadata.get("karar_tarihi"):
            title_parts.append(f"Tarih: {metadata['karar_tarihi']}")
        return " - ".join(title_parts) if title_parts else f"Document {fallback_id}"

    def _semantic_metadata(decision: Any) -> Dict[str, Any]:
        document_id = getattr(decision, "documentId", None)
        metadata = {
            "document_id": document_id,
            "birim_adi": getattr(decision, "birimAdi", None),
            "esas_no": getattr(decision, "esasNo", None),
            "karar_no": getattr(decision, "kararNo", None),
            "karar_tarihi": getattr(decision, "kararTarihiStr", None),
            "court_type": _decision_court_type(decision),
        }
        metadata["title"] = _semantic_title(metadata, document_id)
        metadata["source_url"] = f"https://mevzuat.adalet.gov.tr/ictihat/{document_id}" if document_id else None
        return metadata

    def _semantic_candidate_preview(
        metadata: Dict[str, Any],
        fetch_status: str = "not_attempted",
        preview_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        preview = preview_text[:500] if preview_text else None
        return {
            "document_id": metadata.get("document_id"),
            "title": metadata.get("title") or _semantic_title(metadata, metadata.get("document_id")),
            "birim_adi": metadata.get("birim_adi"),
            "esas_no": metadata.get("esas_no"),
            "karar_no": metadata.get("karar_no"),
            "karar_tarihi": metadata.get("karar_tarihi"),
            "court_type": metadata.get("court_type"),
            "source_url": metadata.get("source_url"),
            "preview_available": bool(preview),
            "preview_text": preview,
            "fetch_status": fetch_status,
        }

    def _semantic_base_diagnostics(
        court_types: List[Any],
        max_candidates: int,
        top_k: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        return {
            "provider": None,
            "embedding_model": None,
            "court_types_requested": [_court_type_value(court_type) for court_type in court_types],
            "max_candidates": max_candidates,
            "top_k": top_k,
            "documents_found": 0,
            "search_attempted": 0,
            "search_succeeded": 0,
            "search_failed": 0,
            "fetch_attempted": 0,
            "fetch_succeeded": 0,
            "failed_fetches": 0,
            "search_ms": 0,
            "fetch_ms_total": 0,
            "embedding_ms": 0,
            "total_ms": 0,
            "timed_out": False,
            "timeout_s": timeout_s,
        }

    def _semantic_response(
        status: str,
        message: str,
        diagnostics: Dict[str, Any],
        start_time: float,
        *,
        candidates_preview: Optional[List[Dict[str, Any]]] = None,
        results: Optional[List[Dict[str, Any]]] = None,
        semantic_ranking_skipped: Optional[bool] = None,
        query: Optional[str] = None,
        initial_keyword: Optional[str] = None,
    ) -> Dict[str, Any]:
        diagnostics["total_ms"] = _semantic_now_ms(start_time)
        response = {
            "status": status,
            "message": message,
            "diagnostics": diagnostics,
        }
        if query is not None:
            response["query"] = query
        if initial_keyword is not None:
            response["initial_keyword"] = initial_keyword
        if results is not None:
            response["results"] = results
        if candidates_preview is not None:
            response["candidates_preview"] = candidates_preview
        if semantic_ranking_skipped is not None:
            response["semantic_ranking_skipped"] = semantic_ranking_skipped
        return response

    def _round_robin_decisions(grouped_decisions: Dict[str, List[Any]], court_types: List[Any], limit: int) -> List[Any]:
        selected = []
        court_order = [_court_type_value(court_type) for court_type in court_types]
        max_group_len = max((len(grouped_decisions.get(court_type, [])) for court_type in court_order), default=0)
        for index in range(max_group_len):
            for court_type in court_order:
                group = grouped_decisions.get(court_type, [])
                if index < len(group):
                    selected.append(group[index])
                    if len(selected) >= limit:
                        return selected
        return selected

    @app.tool(
        description=(
            "Use this to semantically re-rank a small candidate set of Turkish court decision records from Bedesten. "
            "This is not a broad search tool: default scope is Yargıtay + Appeals Court decisions, top_k/max_candidates default to 8, "
            "and results is returned only after embedding re-ranking succeeds. Timeout/error responses use candidates_preview instead. "
            "Set allow_broad_search=True only when intentionally searching 3+ court types; broad or concurrent semantic searches increase partial_timeout risk."
        ),
        annotations={
            "readOnlyHint": True,
            "openWorldHint": True,
            "idempotentHint": True
        }
    )
    async def search_bedesten_semantic(
        initial_keyword: str = Field(..., description="""Bedesten API'den sınırlı aday kararları çekmek için anahtar kelime veya arama ifadesi.
Bu terim sadece aday setini bulur; semantik sıralama query alanına göre yapılır.

ARAMA OPERATÖRLERİ:
• Basit arama: "muvazaa" (kelimeyi içeren kararlar)
• Tam eşleşme: "\"muris muvazaası\"" (tırnak içi aynen aranır)
• AND: "muvazaa AND tapu" (her iki terim zorunlu)
• OR: "ecrimisil OR kira" (en az biri yeterli)
• NOT: "muvazaa NOT miras" (muvazaa içeren ama miras içermeyen)
• Zorunlu: "+muvazaa tapu" (muvazaa zorunlu, tapu opsiyonel)
• Hariç: "muvazaa -miras" (muvazaa içeren, miras hariç)

Tek kelimelik aramalar geçerlidir, ancak çok geniş sonuç üretebilir. Varsayılan max_candidates=8 bu genişliği sınırlar."""),
        query: str = Field(..., description="""Semantik benzerlik için DETAYLI arama sorgusu.
initial_keyword ile bulunan sınırlı aday kararlar bu sorguya göre anlamsal olarak sıralanır.

ÖNEMLİ: Embedding modeli anlamlı cümleler bekler, anahtar kelimeler DEĞİL.
Aradığınız hukuki meseleyi CÜMLE olarak yazın.

DOĞRU KULLANIM:
• "Mirasçının muvazaalı satış işlemine karşı tapu iptali ve tescil davası açması"
• "Taşınmazın fiili kullanımı ve zilyetlik durumunun değerlendirilmesi"
• "İş sözleşmesinin feshinde kıdem tazminatı hesaplama yöntemi"

YANLIŞ KULLANIM:
• "muvazaa tapu iptal" (sadece kelimeler, cümle değil)
• "kıdem tazminat hesap" (bağlamsız kelimeler)"""),
        court_types: List[BedestenCourtTypeEnum] = Field(
            default=["YARGITAYKARARI", "ISTINAFHUKUK"],
            description=(
                "Court types to search. Default is YARGITAYKARARI + ISTINAFHUKUK. "
                "Use allow_broad_search=True when selecting 3+ court types."
            )
        ),
        top_k: int = Field(8, ge=1, le=20, description="Number of semantically ranked results to return (1-20). Must be <= max_candidates."),
        max_candidates: int = Field(8, ge=1, le=20, description="Maximum number of document bodies to fetch and embed (1-20)."),
        allow_broad_search: bool = Field(False, description="Set true to allow searching 3 or more court types; this increases partial_timeout risk.")
    ) -> Dict[str, Any]:
        """
        Perform bounded semantic re-ranking on Bedesten court decision records.
        """
        start_time = time.monotonic()
        timeout_s = _semantic_float_env("SEMANTIC_SEARCH_TIMEOUT_S", 45.0, 15.0, 120.0)
        call_timeout_s = _semantic_float_env("SEMANTIC_CALL_TIMEOUT_S", 12.0, 3.0, 30.0)
        embedding_timeout_s = get_embedding_request_timeout_s()
        if isinstance(court_types, str):
            court_types = [court_types]
        elif isinstance(court_types, tuple):
            court_types = list(court_types)
        elif not isinstance(court_types, list):
            court_types = ["YARGITAYKARARI", "ISTINAFHUKUK"]
        if not isinstance(top_k, int):
            top_k = 8
        if not isinstance(max_candidates, int):
            max_candidates = 8
        if not isinstance(allow_broad_search, bool):
            allow_broad_search = False
        deadline = start_time + timeout_s
        diagnostics = _semantic_base_diagnostics(court_types, max_candidates, top_k, timeout_s)

        logger.info(
            "Semantic search tool called: initial_keyword=%r, top_k=%s, max_candidates=%s, court_types=%s",
            initial_keyword,
            top_k,
            max_candidates,
            [_court_type_value(court_type) for court_type in court_types],
        )

        keyword = (initial_keyword or "").strip()
        if len(keyword) < 2:
            return _semantic_response(
                "validation_error",
                "initial_keyword must contain at least 2 non-whitespace characters.",
                diagnostics,
                start_time,
                query=query,
                initial_keyword=initial_keyword,
            )
        if top_k > max_candidates:
            return _semantic_response(
                "validation_error",
                "top_k must be less than or equal to max_candidates.",
                diagnostics,
                start_time,
                query=query,
                initial_keyword=initial_keyword,
            )
        if len(court_types) >= 3 and not allow_broad_search:
            return _semantic_response(
                "validation_error",
                "Selecting 3 or more court types requires allow_broad_search=True.",
                diagnostics,
                start_time,
                query=query,
                initial_keyword=initial_keyword,
            )

        grouped_decisions: Dict[str, List[Any]] = {}
        search_start = time.monotonic()

        for court_type in court_types:
            remaining = _semantic_remaining_s(deadline)
            if remaining <= 0:
                diagnostics["timed_out"] = True
                break

            court_type_name = _court_type_value(court_type)
            diagnostics["search_attempted"] += 1
            try:
                search_request = BedestenSearchRequest(
                    data=BedestenSearchData(
                        phrase=keyword,
                        itemTypeList=[court_type],
                        pageSize=min(10, max_candidates),
                        pageNumber=1
                    )
                )
                search_results = await asyncio.wait_for(
                    bedesten_client_instance.search_documents(search_request),
                    timeout=min(call_timeout_s, remaining),
                )
                decisions = []
                if search_results.data and search_results.data.emsalKararList:
                    decisions = search_results.data.emsalKararList
                grouped_decisions[court_type_name] = decisions
                diagnostics["search_succeeded"] += 1
                diagnostics["documents_found"] += len(decisions)
                logger.info("Semantic search found %s candidates from %s", len(decisions), court_type_name)
            except asyncio.TimeoutError:
                diagnostics["search_failed"] += 1
                diagnostics["timed_out"] = True
                logger.warning("Semantic search Bedesten search timed out for %s", court_type_name)
            except Exception as e:
                diagnostics["search_failed"] += 1
                logger.warning("Semantic search Bedesten search failed for %s: %s", court_type_name, e)

        diagnostics["search_ms"] = _semantic_now_ms(search_start)
        decisions_to_process = _round_robin_decisions(grouped_decisions, court_types, max_candidates)

        if not decisions_to_process:
            if diagnostics["timed_out"]:
                return _semantic_response(
                    "partial_timeout",
                    "Semantic search timed out before collecting candidate decisions.",
                    diagnostics,
                    start_time,
                    candidates_preview=[],
                    semantic_ranking_skipped=True,
                    query=query,
                    initial_keyword=initial_keyword,
                )
            return _semantic_response(
                "no_results",
                "No documents found matching the initial keyword.",
                diagnostics,
                start_time,
                query=query,
                initial_keyword=initial_keyword,
            )

        documents_data = []
        candidates_preview = [
            _semantic_candidate_preview(_semantic_metadata(decision))
            for decision in decisions_to_process
        ]
        preview_by_id = {preview["document_id"]: preview for preview in candidates_preview}
        processor = DocumentProcessor(chunk_size=1500, chunk_overlap=300)
        fetch_start = time.monotonic()

        for decision in decisions_to_process:
            remaining = _semantic_remaining_s(deadline)
            if remaining <= 0:
                diagnostics["timed_out"] = True
                break

            metadata = _semantic_metadata(decision)
            document_id = metadata["document_id"]
            preview = preview_by_id.get(document_id)
            diagnostics["fetch_attempted"] += 1
            try:
                doc = await asyncio.wait_for(
                    bedesten_client_instance.get_document_as_markdown(document_id),
                    timeout=min(call_timeout_s, remaining),
                )
                markdown_content = doc.markdown_content or ""
                if preview is not None:
                    preview.update(_semantic_candidate_preview(metadata, "fetched", markdown_content))

                if markdown_content:
                    chunks = processor.process_document(
                        document_id=document_id,
                        text=markdown_content,
                        metadata=metadata
                    )
                    if chunks:
                        full_text = " ".join([chunk.text for chunk in chunks])
                        documents_data.append({
                            "id": document_id,
                            "text": full_text[:3000],
                            "metadata": metadata
                        })
                        diagnostics["fetch_succeeded"] += 1
                    else:
                        diagnostics["failed_fetches"] += 1
                        if preview is not None:
                            preview["fetch_status"] = "failed"
                else:
                    diagnostics["failed_fetches"] += 1
                    if preview is not None:
                        preview["fetch_status"] = "failed"
            except asyncio.TimeoutError:
                diagnostics["failed_fetches"] += 1
                diagnostics["timed_out"] = True
                if preview is not None:
                    preview["fetch_status"] = "timeout"
                logger.warning("Semantic search document fetch timed out for %s", document_id)
            except Exception as e:
                diagnostics["failed_fetches"] += 1
                if preview is not None:
                    preview["fetch_status"] = "failed"
                logger.warning("Semantic search document fetch failed for %s: %s", document_id, e)

        diagnostics["fetch_ms_total"] = _semantic_now_ms(fetch_start)

        if diagnostics["timed_out"] or _semantic_remaining_s(deadline) <= 0:
            diagnostics["timed_out"] = True
            return _semantic_response(
                "partial_timeout",
                "Semantic search timed out before embedding re-ranking completed.",
                diagnostics,
                start_time,
                candidates_preview=candidates_preview,
                semantic_ranking_skipped=True,
                query=query,
                initial_keyword=initial_keyword,
            )

        if not documents_data:
            return _semantic_response(
                "embedding_error",
                "No fetched document content could be processed for semantic ranking.",
                diagnostics,
                start_time,
                candidates_preview=candidates_preview,
                query=query,
                initial_keyword=initial_keyword,
            )

        if _semantic_remaining_s(deadline) <= 0:
            diagnostics["timed_out"] = True
            return _semantic_response(
                "partial_timeout",
                "Semantic search timed out before embedding started.",
                diagnostics,
                start_time,
                candidates_preview=candidates_preview,
                semantic_ranking_skipped=True,
                query=query,
                initial_keyword=initial_keyword,
            )

        embedding_started = False
        embedding_start = time.monotonic()
        try:
            embedder = get_embedder()
            diagnostics["provider"] = getattr(
                embedder,
                "provider",
                "local" if is_local_embedding_configured() else "openrouter"
            )
            diagnostics["embedding_model"] = getattr(embedder, "model", None)
            vector_store = VectorStore(dimension=embedder.dimension)

            remaining = _semantic_remaining_s(deadline)
            if remaining <= 0:
                diagnostics["timed_out"] = True
                return _semantic_response(
                    "partial_timeout",
                    "Semantic search timed out before embedding started.",
                    diagnostics,
                    start_time,
                    candidates_preview=candidates_preview,
                    semantic_ranking_skipped=True,
                    query=query,
                    initial_keyword=initial_keyword,
                )
            embedding_started = True
            query_embedding = await asyncio.wait_for(
                asyncio.to_thread(embedder.encode_query, query, "search result"),
                timeout=min(embedding_timeout_s, remaining),
            )

            remaining = _semantic_remaining_s(deadline)
            if remaining <= 0:
                raise asyncio.TimeoutError()

            doc_texts = [doc["text"] for doc in documents_data]
            doc_titles = [doc["metadata"].get("birim_adi", "none") for doc in documents_data]
            doc_embeddings = await asyncio.wait_for(
                asyncio.to_thread(partial(embedder.encode_documents, doc_texts, titles=doc_titles)),
                timeout=min(embedding_timeout_s, remaining),
            )
        except Exception as e:
            diagnostics["embedding_ms"] = _semantic_now_ms(embedding_start)
            if isinstance(e, asyncio.TimeoutError):
                diagnostics["timed_out"] = True
            message = (
                "Embedding request failed or timed out after embedding started."
                if embedding_started
                else "Embedding initialization failed before any embedding request was sent."
            )
            return _semantic_response(
                "embedding_error",
                message,
                diagnostics,
                start_time,
                candidates_preview=candidates_preview,
                query=query,
                initial_keyword=initial_keyword,
            )

        diagnostics["embedding_ms"] = _semantic_now_ms(embedding_start)

        doc_ids = [doc["id"] for doc in documents_data]
        doc_metadatas = [doc["metadata"] for doc in documents_data]
        try:
            vector_store.add_documents(
                ids=doc_ids,
                texts=doc_texts,
                embeddings=doc_embeddings,
                metadata=doc_metadatas
            )

            search_results = vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=0.3
            )

            formatted_results = []
            for doc, score in search_results:
                formatted_results.append({
                    "document_id": doc.id,
                    "title": _semantic_title(doc.metadata, doc.id),
                    "similarity_score": float(score),
                    "preview": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                    "metadata": doc.metadata,
                    "source_url": doc.metadata.get("source_url") or f"https://mevzuat.adalet.gov.tr/ictihat/{doc.id}"
                })
        except Exception as e:
            logger.warning("Semantic search vector ranking failed: %s", e)
            return _semantic_response(
                "embedding_error",
                "Semantic ranking failed after embeddings were generated.",
                diagnostics,
                start_time,
                candidates_preview=candidates_preview,
                query=query,
                initial_keyword=initial_keyword,
            )

        return _semantic_response(
            "success",
            "Semantic search completed successfully.",
            diagnostics,
            start_time,
            results=formatted_results,
            query=query,
            initial_keyword=initial_keyword,
        )


# --- MCP Tools for Sayıştay (Turkish Court of Accounts) ---

# DEACTIVATED TOOL - Use search_sayistay_unified instead
# @app.tool(
#     description="Search Sayıştay Genel Kurul decisions for audit and accountability regulations",
#     annotations={
#         "readOnlyHint": True,
#         "openWorldHint": True,
#         "idempotentHint": True
#     }
# )
# async def search_sayistay_genel_kurul(
#     karar_no: str = Field("", description="Decision number to search for (e.g., '5415')"),
#     karar_ek: str = Field("", description="Decision appendix number (max 99, e.g., '1')"),
#     karar_tarih_baslangic: str = Field("", description="Start date (DD.MM.YYYY)"),
#     karar_tarih_bitis: str = Field("", description="End date (DD.MM.YYYY)"),
#     karar_tamami: str = Field("", description="Full text search"),
#     start: int = Field(0, description="Starting record for pagination (0-based)"),
#     length: int = Field(10, description="Number of records per page (1-100)")
# ) -> GenelKurulSearchResponse:
#     """Search Sayıştay General Assembly decisions."""
#     raise ValueError("This tool is deactivated. Use search_sayistay_unified instead.")

# DEACTIVATED TOOL - Use search_sayistay_unified instead
# @app.tool(
#     description="Search Sayıştay Temyiz Kurulu decisions with chamber filtering and comprehensive criteria",
#     annotations={
#         "readOnlyHint": True,
#         "openWorldHint": True,
#         "idempotentHint": True
#     }
# )
# async def search_sayistay_temyiz_kurulu(
#     ilam_dairesi: DaireEnum = Field("ALL", description="Audit chamber selection"),
#     yili: str = Field("", description="Year (YYYY)"),
#     karar_tarih_baslangic: str = Field("", description="Start date (DD.MM.YYYY)"),
#     karar_tarih_bitis: str = Field("", description="End date (DD.MM.YYYY)"),
#     kamu_idaresi_turu: KamuIdaresiTuruEnum = Field("ALL", description="Public admin type"),
#     ilam_no: str = Field("", description="Audit report number (İlam No, max 50 chars)"),
#     dosya_no: str = Field("", description="File number for the case"),
#     temyiz_tutanak_no: str = Field("", description="Appeals board meeting minutes number"),
#     temyiz_karar: str = Field("", description="Appeals decision text"),
#     web_karar_konusu: WebKararKonusuEnum = Field("ALL", description="Decision subject"),
#     start: int = Field(0, description="Starting record for pagination (0-based)"),
#     length: int = Field(10, description="Number of records per page (1-100)")
# ) -> TemyizKuruluSearchResponse:
#     """Search Sayıştay Appeals Board decisions."""
#     raise ValueError("This tool is deactivated. Use search_sayistay_unified instead.")

# DEACTIVATED TOOL - Use search_sayistay_unified instead
# @app.tool(
#     description="Search Sayıştay Daire decisions with chamber filtering and subject categorization",
#     annotations={
#         "readOnlyHint": True,
#         "openWorldHint": True,
#         "idempotentHint": True
#     }
# )
# async def search_sayistay_daire(
#     yargilama_dairesi: DaireEnum = Field("ALL", description="Chamber selection"),
#     karar_tarih_baslangic: str = Field("", description="Start date (DD.MM.YYYY)"),
#     karar_tarih_bitis: str = Field("", description="End date (DD.MM.YYYY)"),
#     ilam_no: str = Field("", description="Audit report number (İlam No, max 50 chars)"),
#     kamu_idaresi_turu: KamuIdaresiTuruEnum = Field("ALL", description="Public admin type"),
#     hesap_yili: str = Field("", description="Fiscal year"),
#     web_karar_konusu: WebKararKonusuEnum = Field("ALL", description="Decision subject"),
#     web_karar_metni: str = Field("", description="Decision text search"),
#     start: int = Field(0, description="Starting record for pagination (0-based)"),
#     length: int = Field(10, description="Number of records per page (1-100)")
# ) -> DaireSearchResponse:
#     """Search Sayıştay Chamber decisions."""
#     raise ValueError("This tool is deactivated. Use search_sayistay_unified instead.")

# DEACTIVATED TOOL - Use get_sayistay_document_unified instead
# @app.tool(
#     description="Get Sayıştay Genel Kurul decision document in Markdown format",
#     annotations={
#         "readOnlyHint": True,
#         "openWorldHint": False,
#         "idempotentHint": True
#     }
# )
# async def get_sayistay_genel_kurul_document_markdown(
#     decision_id: str = Field(..., description="Decision ID from search_sayistay_genel_kurul results")
# ) -> SayistayDocumentMarkdown:
#     """Get Sayıştay General Assembly decision as Markdown."""
#     raise ValueError("This tool is deactivated. Use get_sayistay_document_unified instead.")

# DEACTIVATED TOOL - Use get_sayistay_document_unified instead
# @app.tool(
#     description="Get Sayıştay Temyiz Kurulu decision document in Markdown format",
#     annotations={
#         "readOnlyHint": True,
#         "openWorldHint": False,
#         "idempotentHint": True
#     }
# )
# async def get_sayistay_temyiz_kurulu_document_markdown(
#     decision_id: str = Field(..., description="Decision ID from search_sayistay_temyiz_kurulu results")
# ) -> SayistayDocumentMarkdown:
#     """Get Sayıştay Appeals Board decision as Markdown."""
#     raise ValueError("This tool is deactivated. Use get_sayistay_document_unified instead.")

# DEACTIVATED TOOL - Use get_sayistay_document_unified instead
# @app.tool(
#     description="Get Sayıştay Daire decision document in Markdown format",
#     annotations={
#         "readOnlyHint": True,
#         "openWorldHint": False,
#         "idempotentHint": True
#     }
# )
# async def get_sayistay_daire_document_markdown(
#     decision_id: str = Field(..., description="Decision ID from search_sayistay_daire results")
# ) -> SayistayDocumentMarkdown:
#     """Get Sayıştay Chamber decision as Markdown."""
#     raise ValueError("This tool is deactivated. Use get_sayistay_document_unified instead.")

# --- UNIFIED MCP Tools for Sayıştay (Turkish Court of Accounts) ---

@app.tool(
    description="Use this when searching Turkish Court of Accounts (Sayıştay) audit decisions. Supports Genel Kurul, Temyiz Kurulu, and Daire decisions.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_sayistay_unified(
    decision_type: Literal["genel_kurul", "temyiz_kurulu", "daire"] = Field(..., description="Decision type: genel_kurul, temyiz_kurulu, or daire"),
    
    # Common pagination parameters
    start: int = Field(0, ge=0, description="Starting record for pagination (0-based)"),
    length: int = Field(10, ge=1, le=100, description="Number of records per page (1-100)"),
    
    # Common search parameters
    karar_tarih_baslangic: str = Field("", description="Start date (DD.MM.YYYY format)"),
    karar_tarih_bitis: str = Field("", description="End date (DD.MM.YYYY format)"),
    kamu_idaresi_turu: Literal["ALL", "Genel Bütçe Kapsamındaki İdareler", "Yüksek Öğretim Kurumları", "Diğer Özel Bütçeli İdareler", "Düzenleyici ve Denetleyici Kurumlar", "Sosyal Güvenlik Kurumları", "Özel İdareler", "Belediyeler ve Bağlı İdareler", "Diğer"] = Field("ALL", description="Public administration type filter"),
    ilam_no: str = Field("", description="Audit report number (İlam No, max 50 chars)"),
    web_karar_konusu: Literal["ALL", "Harcırah Mevzuatı", "İhale Mevzuatı", "İş Mevzuatı", "Personel Mevzuatı", "Sorumluluk ve Yargılama Usulleri", "Vergi Resmi Harç ve Diğer Gelirler", "Çeşitli Konular"] = Field("ALL", description="Decision subject category filter"),
    
    # Genel Kurul specific parameters (ignored for other types)
    karar_no: str = Field("", description="Decision number (genel_kurul only)"),
    karar_ek: str = Field("", description="Decision appendix number (genel_kurul only)"),
    karar_tamami: str = Field("", description="Full text search (genel_kurul only)"),
    
    # Temyiz Kurulu specific parameters (ignored for other types)
    ilam_dairesi: Literal["ALL", "1", "2", "3", "4", "5", "6", "7", "8"] = Field("ALL", description="Audit chamber selection (temyiz_kurulu only)"),
    yili: str = Field("", description="Year (YYYY format, temyiz_kurulu only)"),
    dosya_no: str = Field("", description="File number (temyiz_kurulu only)"),
    temyiz_tutanak_no: str = Field("", description="Appeals board meeting minutes number (temyiz_kurulu only)"),
    temyiz_karar: str = Field("", description="Appeals decision text search (temyiz_kurulu only)"),
    
    # Daire specific parameters (ignored for other types)
    yargilama_dairesi: Literal["ALL", "1", "2", "3", "4", "5", "6", "7", "8"] = Field("ALL", description="Chamber selection (daire only)"),
    hesap_yili: str = Field("", description="Account year (daire only)"),
    web_karar_metni: str = Field("", description="Decision text search (daire only)")
) -> Dict[str, Any]:
    """Search Sayıştay decisions across all three decision types with unified interface."""
    logger.info(f"Tool 'search_sayistay_unified' called with decision_type={decision_type}")

    try:
        search_request = SayistayUnifiedSearchRequest(
            decision_type=decision_type,
            start=start,
            length=length,
            karar_tarih_baslangic=karar_tarih_baslangic,
            karar_tarih_bitis=karar_tarih_bitis,
            kamu_idaresi_turu=kamu_idaresi_turu,
            ilam_no=ilam_no,
            web_karar_konusu=web_karar_konusu,
            karar_no=karar_no,
            karar_ek=karar_ek,
            karar_tamami=karar_tamami,
            ilam_dairesi=ilam_dairesi,
            yili=yili,
            dosya_no=dosya_no,
            temyiz_tutanak_no=temyiz_tutanak_no,
            temyiz_karar=temyiz_karar,
            yargilama_dairesi=yargilama_dairesi,
            hesap_yili=hesap_yili,
            web_karar_metni=web_karar_metni
        )
        result = await sayistay_unified_client_instance.search_unified(search_request)
        return result.model_dump()
    except Exception:
        logger.exception("Error in tool 'search_sayistay_unified'")
        raise

@app.tool(
    description="Use this when retrieving full text of a Sayıştay audit decision. Returns clean Markdown format.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def get_sayistay_document_unified(
    decision_id: str = Field(..., description="Decision ID from search_sayistay_unified results"),
    decision_type: Literal["genel_kurul", "temyiz_kurulu", "daire"] = Field(..., description="Decision type: genel_kurul, temyiz_kurulu, or daire")
) -> Dict[str, Any]:
    """Get Sayıştay decision document as Markdown for any decision type."""
    logger.info(f"Tool 'get_sayistay_document_unified' called for ID: {decision_id}, type: {decision_type}")

    if not decision_id or not decision_id.strip():
        raise ValueError("Decision ID must be a non-empty string.")

    try:
        result = await sayistay_unified_client_instance.get_document_unified(decision_id, decision_type)
        return result.model_dump()
    except Exception:
        logger.exception("Error in tool 'get_sayistay_document_unified'")
        raise

# --- Application Shutdown Handling ---
def perform_cleanup():
    logger.info("MCP Server performing cleanup...")
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed(): 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError: 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    clients_to_close = [
        globals().get('yargitay_client_instance'),
        globals().get('danistay_client_instance'),
        globals().get('emsal_client_instance'),
        globals().get('uyusmazlik_client_instance'),
        globals().get('anayasa_norm_client_instance'),
        globals().get('anayasa_bireysel_client_instance'),
        globals().get('anayasa_unified_client_instance'),
        globals().get('kik_v2_client_instance'),
        globals().get('rekabet_client_instance'),
        globals().get('bedesten_client_instance'),
        globals().get('sayistay_client_instance'),
        globals().get('sayistay_unified_client_instance'),
        globals().get('kvkk_client_instance'),
        globals().get('bddk_client_instance'),
        globals().get('gib_client_instance'),
        globals().get('sigorta_tahkim_client_instance')
    ]
    async def close_all_clients_async():
        tasks = []
        for client_instance in clients_to_close:
            if client_instance and hasattr(client_instance, 'close_client_session') and callable(client_instance.close_client_session):
                logger.info(f"Scheduling close for client session: {client_instance.__class__.__name__}")
                tasks.append(client_instance.close_client_session())
        # Close health check client if it was created
        global _health_check_client
        if _health_check_client is not None:
            logger.info("Closing health check HTTP client")
            tasks.append(_health_check_client.aclose())
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client_name = "Unknown Client"
                    if i < len(clients_to_close) and clients_to_close[i] is not None:
                        client_name = clients_to_close[i].__class__.__name__
                    logger.error(f"Error closing client {client_name}: {result}")
    try:
        if loop.is_running(): 
            asyncio.ensure_future(close_all_clients_async(), loop=loop)
            logger.info("Client cleanup tasks scheduled on running event loop.")
        else:
            loop.run_until_complete(close_all_clients_async())
            logger.info("Client cleanup tasks completed via run_until_complete.")
    except Exception as e: 
        logger.error(f"Error during atexit cleanup execution: {e}", exc_info=True)
    logger.info("MCP Server atexit cleanup process finished.")

atexit.register(perform_cleanup)


def get_or_create_health_check_client() -> httpx.AsyncClient:
    """Get or create a reusable HTTP client for health checks."""
    global _health_check_client
    if _health_check_client is None:
        _health_check_client = httpx.AsyncClient(
            timeout=10.0,
            verify=False,
            follow_redirects=True
        )
    return _health_check_client


# --- Health Check Tools ---
@app.tool(
    description="Use this when checking if Turkish legal database servers are online and responding.",
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True
    }
)
async def check_government_servers_health() -> Dict[str, Any]:
    """Check health status of Turkish government legal database servers."""
    logger.info("Health check tool called for government servers")
    
    health_results = {}
    
    # Check Yargıtay server
    try:
        yargitay_payload = {
            "data": {
                "aranan": "karar",
                "arananKelime": "karar", 
                "pageSize": 10,
                "pageNumber": 1
            }
        }
        
        async with httpx.AsyncClient(
            headers={
                "Accept": "*/*",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "Connection": "keep-alive",
                "Content-Type": "application/json; charset=UTF-8",
                "Origin": "https://karararama.yargitay.gov.tr",
                "Referer": "https://karararama.yargitay.gov.tr/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors", 
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
                "X-Requested-With": "XMLHttpRequest"
            },
            timeout=30.0,
            verify=False
        ) as client:
            response = await client.post(
                "https://karararama.yargitay.gov.tr/aramalist",
                json=yargitay_payload
            )
        
        if response.status_code == 200:
            response_data = response.json()
            records_total = response_data.get("data", {}).get("recordsTotal", 0)
            
            if records_total > 0:
                health_results["yargitay"] = {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                health_results["yargitay"] = {
                    "status": "unhealthy", 
                    "reason": "recordsTotal is 0 or missing",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
        else:
            health_results["yargitay"] = {
                "status": "unhealthy", 
                "reason": f"HTTP {response.status_code}",
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
        
    except Exception as e:
        health_results["yargitay"] = {
            "status": "unhealthy",
            "reason": f"Connection error: {str(e)}"
        }
    
    # Check Bedesten API server
    try:
        bedesten_payload = {
            "data": {
                "pageSize": 5,
                "pageNumber": 1,
                "itemTypeList": ["YARGITAYKARARI"], 
                "phrase": "karar",
                "sortFields": ["KARAR_TARIHI"],
                "sortDirection": "desc"
            },
            "applicationName": "UyapMevzuat",
            "paging": True
        }
        
        client = get_or_create_health_check_client()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 Health Check"
        }
        
        response = await client.post(
            "https://bedesten.adalet.gov.tr/emsal-karar/searchDocuments",
            json=bedesten_payload,
            headers=headers
        )
        
        if response.status_code == 200:
            response_data = response.json()
            logger.debug(f"Bedesten API response: {response_data}")
            if response_data and isinstance(response_data, dict):
                data_section = response_data.get("data")
                if data_section and isinstance(data_section, dict):
                    total_found = data_section.get("total", 0)
                else:
                    total_found = 0
            else:
                total_found = 0
            
            if total_found > 0:
                health_results["bedesten"] = {
                    "status": "healthy", 
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                health_results["bedesten"] = {
                    "status": "unhealthy",
                    "reason": "total is 0 or missing in data field",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
        else:
            health_results["bedesten"] = {
                "status": "unhealthy",
                "reason": f"HTTP {response.status_code}",
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
        
    except Exception as e:
        health_results["bedesten"] = {
            "status": "unhealthy", 
            "reason": f"Connection error: {str(e)}"
        }
    
    # Overall health assessment
    healthy_servers = sum(1 for server in health_results.values() if server["status"] == "healthy")
    total_servers = len(health_results)
    
    overall_status = "healthy" if healthy_servers == total_servers else "degraded" if healthy_servers > 0 else "unhealthy"
    
    return {
        "overall_status": overall_status,
        "healthy_servers": healthy_servers,
        "total_servers": total_servers,
        "servers": health_results,
        "check_timestamp": f"{__import__('datetime').datetime.now().isoformat()}"
    }

# --- MCP Tools for KVKK ---
@app.tool(
    description="Use this when searching Turkish data protection (KVKK/GDPR equivalent) decisions. For privacy, consent, and data breach cases.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_kvkk_decisions(
    keywords: str = Field(..., description="Turkish keywords. Supports +required -excluded \"exact phrase\" operators"),
    page: int = Field(1, ge=1, le=50, description="Page number for results (1-50)."),
    # pageSize: int = Field(10, ge=1, le=20, description="Number of results per page (1-20).")
) -> Dict[str, Any]:
    """Search function for legal decisions."""
    logger.info(f"KVKK search tool called with keywords: {keywords}")

    pageSize = 10  # Default value

    search_request = KvkkSearchRequest(
        keywords=keywords,
        page=page,
        pageSize=pageSize
    )

    try:
        result = await kvkk_client_instance.search_decisions(search_request)
        logger.info(f"KVKK search completed. Found {len(result.decisions)} decisions on page {page}")
        return result.model_dump()
    except Exception as e:
        logger.exception(f"Error in KVKK search: {e}")
        # Return empty result on error
        return KvkkSearchResult(
            decisions=[],
            total_results=0,
            page=page,
            pageSize=pageSize,
            query=keywords
        ).model_dump()

@app.tool(
    description="Use this when retrieving full text of a KVKK data protection decision. Returns paginated Markdown with metadata.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def get_kvkk_document_markdown(
    decision_url: str = Field(..., description="KVKK decision URL from search results"),
    page_number: int = Field(1, ge=1, description="Page number for paginated Markdown content (1-indexed, accepts int). Default is 1 (first 5,000 characters).")
) -> Dict[str, Any]:
    """Get KVKK decision as paginated Markdown."""
    logger.info(f"KVKK document retrieval tool called for URL: {decision_url}")

    if not decision_url or not decision_url.strip():
        return KvkkDocumentMarkdown(
            source_url=HttpUrl("https://www.kvkk.gov.tr"),
            title=None,
            decision_date=None,
            decision_number=None,
            subject_summary=None,
            markdown_chunk=None,
            current_page=page_number or 1,
            total_pages=0,
            is_paginated=False,
            error_message="Decision URL is required and cannot be empty."
        ).model_dump()
    
    try:
        # Validate URL format
        if not decision_url.startswith("https://www.kvkk.gov.tr/"):
            return KvkkDocumentMarkdown(
                source_url=HttpUrl(decision_url),
                title=None,
                decision_date=None,
                decision_number=None,
                subject_summary=None,
                markdown_chunk=None,
                current_page=page_number or 1,
                total_pages=0,
                is_paginated=False,
                error_message="Invalid KVKK decision URL format. URL must start with https://www.kvkk.gov.tr/"
            ).model_dump()

        result = await kvkk_client_instance.get_decision_document(decision_url, page_number or 1)
        logger.info(f"KVKK document retrieved successfully. Page {result.current_page}/{result.total_pages}, Content length: {len(result.markdown_chunk) if result.markdown_chunk else 0}")
        return result.model_dump()
        
    except Exception as e:
        logger.exception(f"Error retrieving KVKK document: {e}")
        return KvkkDocumentMarkdown(
            source_url=HttpUrl(decision_url),
            title=None,
            decision_date=None,
            decision_number=None,
            subject_summary=None,
            markdown_chunk=None,
            current_page=page_number or 1,
            total_pages=0,
            is_paginated=False,
            error_message=f"Error retrieving KVKK document: {str(e)}"
        ).model_dump()

# --- MCP Tools for BDDK (Banking Regulation Authority) ---
@app.tool(
    description="Use this when searching Turkish banking regulation (BDDK) decisions. For banking licenses, fintech, and payment services.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_bddk_decisions(
    keywords: str = Field(..., description="Search keywords in Turkish"),
    page: int = Field(1, ge=1, description="Page number")
    # pageSize: int = Field(10, ge=1, le=50, description="Results per page")
) -> dict:
    """Search BDDK banking regulation and supervision decisions."""
    logger.info(f"BDDK search tool called with keywords: {keywords}, page: {page}")
    
    pageSize = 10  # Default value
    
    try:
        search_request = BddkSearchRequest(
            keywords=keywords,
            page=page,
            pageSize=pageSize
        )
        
        result = await bddk_client_instance.search_decisions(search_request)
        logger.info(f"BDDK search completed. Found {len(result.decisions)} decisions on page {page}")
        
        return {
            "decisions": [
                {
                    "title": dec.title,
                    "document_id": dec.document_id,
                    "content": dec.content
                }
                for dec in result.decisions
            ],
            "total_results": result.total_results,
            "page": result.page,
            "pageSize": result.pageSize
        }
    
    except Exception as e:
        logger.exception(f"Error searching BDDK decisions: {e}")
        return {
            "decisions": [],
            "total_results": 0,
            "page": page,
            "pageSize": pageSize,
            "error": str(e)
        }

@app.tool(
    description="Use this when retrieving full text of a BDDK banking regulation decision. Returns paginated Markdown format.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def get_bddk_document_markdown(
    document_id: str = Field(..., description="BDDK document ID (e.g., '310')"),
    page_number: int = Field(1, ge=1, description="Page number")
) -> dict:
    """Retrieve BDDK decision document in Markdown format."""
    logger.info(f"BDDK document retrieval tool called for ID: {document_id}, page: {page_number}")
    
    if not document_id or not document_id.strip():
        return {
            "document_id": document_id,
            "markdown_content": "",
            "page_number": page_number,
            "total_pages": 0,
            "error": "Document ID is required"
        }
    
    try:
        result = await bddk_client_instance.get_document_markdown(document_id, page_number)
        logger.info(f"BDDK document retrieved successfully. Page {result.page_number}/{result.total_pages}")
        
        return {
            "document_id": result.document_id,
            "markdown_content": result.markdown_content,
            "page_number": result.page_number,
            "total_pages": result.total_pages
        }
        
    except Exception as e:
        logger.exception(f"Error retrieving BDDK document: {e}")
        return {
            "document_id": document_id,
            "markdown_content": "",
            "page_number": page_number,
            "total_pages": 0,
            "error": str(e)
        }

# --- MCP Tools for GİB (Gelir İdaresi Başkanlığı / Revenue Administration) Özelgeler ---
@app.tool(
    description=(
        "Search Turkish GİB özelge records (Revenue Administration tax rulings) - 18k+ rulings on VAT, "
        "income tax, corporate tax, stamp duty interpretations. kanunNo filters the related law number; "
        "returned documents are özelge/tax ruling records."
    ),
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_gib_ozelge(
    keywords: str = Field("", description="Turkish keywords searched in title, kanunNo and description (e.g., 'KDV oranı', 'kurumlar vergisi istisna')"),
    ozelgeNo: str = Field("", description="Exact özelge reference number (e.g., 'E-40247694-130-15524')"),
    kanunNo: str = Field("", description="Related law number filter for özelge records, e.g. '3065' for KDV, '193' for Gelir Vergisi"),
    ozelgeStartDate: str = Field("", description="Start date YYYY-MM-DD (e.g., '2024-01-01') or full ISO 8601"),
    ozelgeEndDate: str = Field("", description="End date YYYY-MM-DD (e.g., '2024-12-31') or full ISO 8601"),
    page: int = Field(1, ge=1, description="Page number (1-indexed)"),
    pageSize: int = Field(10, ge=1, le=50, description="Results per page (1-50)")
) -> dict:
    """Search GİB özelgeler (Turkish Revenue Administration tax rulings)."""
    logger.info(
        f"GİB search tool called with keywords='{keywords}', ozelgeNo='{ozelgeNo}', "
        f"kanunNo='{kanunNo}', start={ozelgeStartDate}, end={ozelgeEndDate}, "
        f"page={page}, pageSize={pageSize}"
    )

    try:
        search_request = GibSearchRequest(
            keywords=keywords,
            ozelgeNo=ozelgeNo,
            kanunNo=kanunNo,
            ozelgeStartDate=ozelgeStartDate,
            ozelgeEndDate=ozelgeEndDate,
            page=page,
            pageSize=pageSize,
        )
        result = await gib_client_instance.search_ozelge(search_request)
        logger.info(
            f"GİB search completed. Found {len(result.ozelgeler)} rulings on page {page} "
            f"(total {result.total_results})"
        )
        return result.model_dump()
    except Exception as e:
        logger.exception(f"Error searching GİB özelgeler: {e}")
        return GibSearchResult(
            ozelgeler=[],
            total_results=0,
            total_pages=0,
            current_page=page,
            page_size=pageSize,
        ).model_dump()


@app.tool(
    description="Retrieve full text of a GİB özelge (tax ruling) by numeric ID. Returns paginated Markdown (5000-char chunks) with title, reference number, date and law metadata.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def get_gib_ozelge_document_markdown(
    ozelge_id: int = Field(..., ge=1, description="Numeric özelge ID from search results (e.g., 38849)"),
    page_number: int = Field(1, ge=1, description="Page number for paginated Markdown (1-indexed)")
) -> dict:
    """Retrieve a GİB özelge document in paginated Markdown format."""
    logger.info(f"GİB document retrieval tool called for id={ozelge_id}, page={page_number}")

    try:
        result = await gib_client_instance.get_ozelge_document(ozelge_id, page_number)
        logger.info(
            f"GİB document retrieved. id={ozelge_id} page={result.current_page}/{result.total_pages}"
        )
        return result.model_dump()
    except Exception as e:
        logger.exception(f"Error retrieving GİB document: {e}")
        return GibDocumentMarkdown(
            ozelge_id=ozelge_id,
            current_page=page_number,
            total_pages=0,
            is_paginated=False,
            error_message=str(e),
        ).model_dump()


# --- MCP Tools for Sigorta Tahkim Komisyonu (Insurance Arbitration Commission) ---
@app.tool(
    description="Search Sigorta Tahkim Komisyonu (Insurance Arbitration Commission) decisions from Hakem Karar Dergisi journals (64 issues, 2010-2025). Covers insurance disputes: traffic, health, fire, DASK, life insurance.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search_sigorta_tahkim_decisions(
    keywords: str = Field(..., description="Search keywords in Turkish (e.g., 'trafik sigortası', 'kasko', 'DASK')"),
    page: int = Field(1, ge=1, description="Page number")
) -> dict:
    """Search Sigorta Tahkim Komisyonu insurance arbitration decisions."""
    logger.info(f"Sigorta Tahkim search tool called with keywords: {keywords}, page: {page}")

    pageSize = 10

    try:
        search_request = SigortaTahkimSearchRequest(
            keywords=keywords,
            page=page,
            pageSize=pageSize
        )

        result = await sigorta_tahkim_client_instance.search_decisions(search_request)
        logger.info(f"Sigorta Tahkim search completed. Found {len(result.decisions)} results on page {page}")

        return {
            "decisions": [
                {
                    "title": dec.title,
                    "document_id": dec.document_id,
                    "content": dec.content,
                    "url": dec.url
                }
                for dec in result.decisions
            ],
            "total_results": result.total_results,
            "page": result.page,
            "pageSize": result.pageSize
        }

    except Exception as e:
        logger.exception(f"Error searching Sigorta Tahkim decisions: {e}")
        return {
            "decisions": [],
            "total_results": 0,
            "page": page,
            "pageSize": pageSize,
            "error": str(e)
        }

@app.tool(
    description="Retrieve full PDF content of a Sigorta Tahkim Komisyonu Hakem Karar Dergisi issue by number. Returns paginated Markdown. Issues 1-64 available (2010-2025).",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def get_sigorta_tahkim_document_markdown(
    issue_number: str = Field(..., description="Journal issue number (1-64, e.g., '64')"),
    page_number: int = Field(1, ge=1, description="Page number for paginated content")
) -> dict:
    """Retrieve Sigorta Tahkim journal issue PDF as paginated Markdown."""
    logger.info(f"Sigorta Tahkim document retrieval for issue: {issue_number}, page: {page_number}")

    if not issue_number or not issue_number.strip():
        return {
            "document_id": issue_number,
            "markdown_content": "",
            "page_number": page_number,
            "total_pages": 0,
            "source_url": "",
            "error": "Issue number is required"
        }

    try:
        result = await sigorta_tahkim_client_instance.get_document_markdown(issue_number, page_number)
        logger.info(f"Sigorta Tahkim document retrieved. Page {result.page_number}/{result.total_pages}")

        return {
            "document_id": result.document_id,
            "markdown_content": result.markdown_content,
            "page_number": result.page_number,
            "total_pages": result.total_pages,
            "source_url": result.source_url
        }

    except Exception as e:
        logger.exception(f"Error retrieving Sigorta Tahkim document: {e}")
        return {
            "document_id": issue_number,
            "markdown_content": "",
            "page_number": page_number,
            "total_pages": 0,
            "source_url": "",
            "error": str(e)
        }

@app.tool(
    description="Search within a specific Sigorta Tahkim Komisyonu journal issue for keywords. Downloads the PDF, splits into individual decisions, and returns matching decisions with excerpts sorted by relevance.",
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True
    }
)
async def search_within_sigorta_tahkim_issue(
    issue_number: str = Field(..., description="Journal issue number (1-64, e.g., '64')"),
    keyword: str = Field(..., description="Search keyword in Turkish (e.g., 'trafik kazası', 'tazminat')"),
    max_results: int = Field(10, ge=1, le=25, description="Max matching decisions to return")
) -> dict:
    """Search for keywords within a specific Sigorta Tahkim journal issue's decisions."""
    logger.info(f"Sigorta Tahkim search_within called: issue={issue_number}, keyword={keyword}")

    if not issue_number or not issue_number.strip():
        return {"issue_number": issue_number, "keyword": keyword, "matches": [], "error": "Issue number is required"}
    if not keyword or not keyword.strip():
        return {"issue_number": issue_number, "keyword": keyword, "matches": [], "error": "Keyword is required"}

    try:
        result = await sigorta_tahkim_client_instance.search_within_issue(
            issue_number, keyword, max_results
        )
        logger.info(
            f"Sigorta Tahkim search_within completed: "
            f"{result.matching_decisions}/{result.total_decisions} decisions match"
        )

        return {
            "issue_number": result.issue_number,
            "keyword": result.keyword,
            "total_decisions": result.total_decisions,
            "matching_decisions": result.matching_decisions,
            "matches": [
                {
                    "decision_header": m.decision_header,
                    "relevance_score": m.relevance_score,
                    "excerpt": m.excerpt,
                    "body_length": m.body_length
                }
                for m in result.matches
            ]
        }

    except Exception as e:
        logger.exception(f"Error in search_within Sigorta Tahkim: {e}")
        return {
            "issue_number": issue_number,
            "keyword": keyword,
            "total_decisions": 0,
            "matching_decisions": 0,
            "matches": [],
            "error": str(e)
        }

# --- ChatGPT Deep Research Compatible Tools ---

def build_bedesten_title(decision: Any, court_name: str) -> str:
    """Build a compact title from Bedesten search metadata without fetching the document."""
    title_parts = [court_name]
    if getattr(decision, "birimAdi", None):
        title_parts.append(str(decision.birimAdi))
    if getattr(decision, "esasNo", None):
        title_parts.append(f"Esas: {decision.esasNo}")
    if getattr(decision, "kararNo", None):
        title_parts.append(f"Karar: {decision.kararNo}")
    if getattr(decision, "kararTarihiStr", None):
        title_parts.append(f"Tarih: {decision.kararTarihiStr}")
    return " - ".join(title_parts) if title_parts else f"{court_name} - Document {decision.documentId}"


def build_bedesten_metadata_preview(decision: Any, court_name: str) -> str:
    """Return Deep Research preview text using only search-result metadata."""
    preview_parts = [f"Kaynak: {court_name}"]
    if getattr(decision, "birimAdi", None):
        preview_parts.append(f"Daire/Kurul: {decision.birimAdi}")
    if getattr(decision, "esasNo", None):
        preview_parts.append(f"Esas No: {decision.esasNo}")
    if getattr(decision, "kararNo", None):
        preview_parts.append(f"Karar No: {decision.kararNo}")
    if getattr(decision, "kararTarihiStr", None):
        preview_parts.append(f"Karar Tarihi: {decision.kararTarihiStr}")
    preview_parts.append("Tam metin için fetch aracını bu sonucun id değeriyle çağırın.")
    return ". ".join(preview_parts)


@app.tool(
    description=(
        "Only for ChatGPT Deep Research. Searches Bedesten-supported Turkish court databases and returns "
        "OpenAI Deep Research compatible results (id, title, text, url). For regular MCP use, prefer "
        "search_bedesten_unified. This tool does not fetch document bodies during search so one query stays "
        "within Bedesten upstream rate limits; call fetch only for selected result IDs."
    ),
    annotations={
        "readOnlyHint": True,
        "openWorldHint": True,
        "idempotentHint": True
    }
)
async def search(
    query: str = Field(..., description="Turkish search query")
) -> Dict[str, Any]:
    """
    Bedesten API search tool for ChatGPT Deep Research compatibility.
    
    This tool searches Turkish legal databases via the unified Bedesten API.
    It supports advanced search operators and covers all major court types.
    
    USAGE RESTRICTION: Only for ChatGPT Deep Research workflows.
    For regular legal research, use search_bedesten_unified with specific court types.
    
    Returns:
    Object with "results" field containing a list of documents with id, title, text preview, and url
    as required by ChatGPT Deep Research specification.
    """
    logger.info(f"ChatGPT Deep Research search tool called with query: {query}")
    
    results = []
    
    try:
        # Search all court types via unified Bedesten API
        court_types = [
            ("YARGITAYKARARI", "Yargıtay"),
            ("DANISTAYKARAR", "Danıştay"),
            ("YERELHUKUK", "Yerel Hukuk Mahkemesi"),
            ("ISTINAFHUKUK", "İstinaf Hukuk Mahkemesi"),
            ("KYB", "Kanun Yararına Bozma")
        ]
        
        for item_type, court_name in court_types:
            try:
                search_results = await bedesten_client_instance.search_documents(
                    BedestenSearchRequest(
                        data=BedestenSearchData(
                            phrase=query,  # Use query as-is to support both regular and exact phrase searches
                            itemTypeList=[item_type],
                            pageSize=5,
                            pageNumber=1
                        )
                    )
                )
                
                # Handle potential None data
                if search_results.data is None:
                    logger.warning(f"No data returned from Bedesten API for {court_name}")
                    continue
                
                # Add results from metadata only. Fetching every document preview
                # would turn one Deep Research search into ~30 Bedesten requests.
                for decision in search_results.data.emsalKararList[:5]:
                    results.append({
                        "id": decision.documentId,
                        "title": build_bedesten_title(decision, court_name),
                        "text": build_bedesten_metadata_preview(decision, court_name),
                        "url": f"https://mevzuat.adalet.gov.tr/ictihat/{decision.documentId}"
                    })
                    
                if search_results.data:
                    logger.info(f"Found {len(search_results.data.emsalKararList)} results from {court_name}")
                else:
                    logger.info(f"Found 0 results from {court_name} (no data returned)")
                
            except Exception as e:
                logger.warning(f"Bedesten API search error for {court_name}: {e}")
        
        # Comment out other API implementations for ChatGPT Deep Research
        """
        # Other API implementations disabled for ChatGPT Deep Research
        # These are available through specific court tools:
        
        # Yargıtay Official API - use search_yargitay_detailed instead
        # Danıştay Official API - use search_danistay_by_keyword instead  
        # Constitutional Court - use search_anayasa_norm_denetimi_decisions instead
        # Competition Authority - use search_rekabet_kurumu_decisions instead
        # Public Procurement Authority - use search_kik_v2_decisions instead
        # Court of Accounts - use search_sayistay_* tools instead
        # UYAP Emsal - use search_emsal_detailed_decisions instead
        # Jurisdictional Disputes Court - use search_uyusmazlik_decisions instead
        """
        
        logger.info(f"ChatGPT Deep Research search completed. Found {len(results)} results via Bedesten API.")
        return {
            "results": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "text": item["text"],
                    "url": item["url"]
                }
                for item in results
            ]
        }
        
    except Exception:
        logger.exception("Error in ChatGPT Deep Research search tool")
        # Return partial results if any were found
        if results:
            return {
                "results": [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "text": item["text"],
                        "url": item["url"]
                    }
                    for item in results
                ]
            }
        raise

@app.tool(
    description=(
        "Only for ChatGPT Deep Research. Retrieves one Turkish legal document by numeric Bedesten ID. "
        "For regular MCP use, prefer get_bedesten_document_markdown. This performs one Bedesten document request "
        "and avoids an extra metadata lookup to respect upstream rate limits."
    ),
    annotations={
        "readOnlyHint": True,
        "openWorldHint": False,  # Retrieves specific documents, not exploring
        "idempotentHint": True
    }
)
async def fetch(
    id: str = Field(..., description="Document identifier from search results (numeric only)")
) -> Dict[str, Any]:
    """
    Bedesten API fetch tool for ChatGPT Deep Research compatibility.
    
    Retrieves the full text content of Turkish legal documents via unified Bedesten API.
    Converts documents from HTML/PDF to clean Markdown format.
    
    USAGE RESTRICTION: Only for ChatGPT Deep Research workflows.
    For regular legal research, use specific court document tools.
    
    Input Format:
    - id: Numeric document identifier from search results (e.g., "730113500", "71370900")
    
    Returns:
    Single object with numeric id, title, text (full Markdown content), mevzuat.adalet.gov.tr url, and metadata fields
    as required by ChatGPT Deep Research specification.
    """
    logger.info(f"ChatGPT Deep Research fetch tool called for document ID: {id}")
    
    if not id or not id.strip():
        raise ValueError("Document ID must be a non-empty string")
    
    try:
        # Use the numeric ID directly with Bedesten API
        doc = await bedesten_client_instance.get_document_as_markdown(id)
        
        title = f"Turkish Legal Document {id}"
        if doc.markdown_content:
            for line in doc.markdown_content.splitlines():
                cleaned_line = line.strip().lstrip("#").strip()
                if cleaned_line:
                    title = cleaned_line[:160]
                    break
        
        return {
            "id": id,
            "title": title,
            "text": doc.markdown_content,
            "url": f"https://mevzuat.adalet.gov.tr/ictihat/{id}",
            "metadata": {
                "database": "Turkish Legal Database via Bedesten API",
                "document_id": id,
                "source_url": doc.source_url,
                "mime_type": doc.mime_type,
                "api_source": "Bedesten Unified API",
                "chatgpt_deep_research": True,
                "rate_limit_optimized": True
            }
        }
        
        # Comment out other API implementations for ChatGPT Deep Research
        """
        # Other API implementations disabled for ChatGPT Deep Research
        # These are available through specific court document tools:
        
        elif id.startswith("yargitay_"):
            # Yargıtay Official API - use get_yargitay_document_markdown instead
            doc_id = id.replace("yargitay_", "")
            doc = await yargitay_client_instance.get_decision_document_as_markdown(doc_id)
            
        elif id.startswith("danistay_"):
            # Danıştay Official API - use get_danistay_document_markdown instead
            doc_id = id.replace("danistay_", "")
            doc = await danistay_client_instance.get_decision_document_as_markdown(doc_id)
            
        elif id.startswith("anayasa_"):
            # Constitutional Court - use get_anayasa_norm_denetimi_document_markdown instead
            doc_id = id.replace("anayasa_", "")
            doc = await anayasa_norm_client_instance.get_decision_document_as_markdown(...)
            
        elif id.startswith("rekabet_"):
            # Competition Authority - use get_rekabet_kurumu_document instead
            doc_id = id.replace("rekabet_", "")
            doc = await rekabet_client_instance.get_decision_document(...)
            
        elif id.startswith("kik_"):
            # Public Procurement Authority - use get_kik_decision_document_as_markdown instead
            doc_id = id.replace("kik_", "")
            doc = await kik_client_instance.get_decision_document_as_markdown(doc_id)
            
        elif id.startswith("local_"):
            # This was already using Bedesten API, but deprecated for ChatGPT Deep Research
            doc_id = id.replace("local_", "")
            doc = await bedesten_client_instance.get_document_as_markdown(doc_id)
        """
        
    except Exception:
        logger.exception(f"Error fetching ChatGPT Deep Research document {id}")
        raise

# --- Token Metrics Tool Removed for Optimization ---

def main():
    # Initialize the app properly with create_app()
    global app
    app = create_app()

    logger.info(f"Starting {app.name} server via main() function...")
    # logger.info(f"Logs will be written to: {LOG_FILE_PATH}")  # File logging disabled

    try:
        app.run()
    except KeyboardInterrupt: 
        logger.info("Server shut down by user (KeyboardInterrupt).")
    except Exception: 
        logger.exception("Server failed to start or crashed.")
    finally:
        logger.info(f"{app.name} server has shut down.")

if __name__ == "__main__": 
    main()
