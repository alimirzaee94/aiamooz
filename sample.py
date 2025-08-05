#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Educational RAG System v5.1 -
==============================================

A sophisticated multi-agent RAG system with intelligent orchestration,
context-aware responses, narrative generation capabilities, and casual chat support.

Key Features:
- Master Orchestrator for intelligent response coordination
- Context-aware narrative generation with pedagogical reasoning
- Dynamic response strategies based on query analysis
- Cross-page relationship detection and integration
- Advanced memory system with graph-based knowledge representation
- Casual chat mode for non-educational conversations
- Enhanced Persian number recognition and normalization
- Improved query understanding and response accuracy
- Production-grade error handling and recovery
- Real-time performance optimization

Author: @eArash
Version: 5.1.0
Date: 7/30/2025
"""

import os
import sys
import re
import json
import time
import asyncio
import logging
import hashlib
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Annotated, TypedDict, Sequence, Literal
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
from enum import Enum
import warnings
from functools import lru_cache, wraps
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# External imports
try:
    from openai import AsyncOpenAI
    import faiss
    from sentence_transformers import CrossEncoder
    from rank_bm25 import BM25Okapi
    import tiktoken
    import nltk
    from nltk.tokenize import word_tokenize
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install openai faiss-cpu sentence-transformers rank-bm25 tiktoken nltk")
    sys.exit(1)

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, validator
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

# Download NLTK data
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


# ==================== Configuration ====================
@dataclass
class RAGConfig:
    """Enhanced configuration for RAG system v5.1"""
    # OpenAI settings
    openai_api_key: str
    primary_model: str = "gpt-4.1"
    fast_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"

    # Model parameters
    temperature: float = 0.3
    max_tokens: int = 6000
    embedding_dimension: int = 3072
    request_timeout: int = 30

    # System behavior
    enable_orchestrator: bool = True
    enable_narrative_mode: bool = True
    enable_context_analysis: bool = True
    enable_quality_assurance: bool = True
    enable_casual_chat: bool = True  # NEW: Enable casual chat mode

    # Enhanced system features
    enable_universal_intelligence: bool = True
    enable_advanced_memory: bool = True
    enable_emotional_intelligence: bool = True
    enable_greeting_module: bool = True

    # Search settings
    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    final_top_k: int = 5
    min_relevance_score: float = 0.7

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Response generation
    response_strategies: List[str] = field(default_factory=lambda: [
        "direct_answer", "narrative_explanation", "comparative_analysis",
        "historical_context", "pedagogical_guidance", "casual_conversation"  # NEW: Added casual
    ])

    # Response modules
    response_modules: List[str] = field(default_factory=lambda: [
        "greeting", "content_delivery", "problem_solving", "summarization",
        "conceptual_learning", "content_discovery", "emotional_support",
        "casual_chat"  # NEW: Added casual chat module
    ])

    # Caching and performance
    enable_cache: bool = True
    cache_ttl: int = 3600
    max_concurrent_requests: int = 10
    browser_cache_enabled: bool = True

    # Quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "minimum": 0.7,
        "target": 0.85,
        "approval": 0.75
    })

    # Configuration paths
    prompts_config_path: str = "prompts_config.json"

    # Persian numbers - NEW: Enhanced Persian number recognition
    persian_numbers: Dict[str, str] = field(default_factory=lambda: {
        'یک': '1', 'دو': '2', 'سه': '3', 'چهار': '4', 'پنج': '5',
        'شش': '6', 'شیش': '6', 'هفت': '7', 'هشت': '8', 'نه': '9', 'ده': '10',
        'یازده': '11', 'دوازده': '12', 'سیزده': '13', 'چهارده': '14', 'پانزده': '15',
        'شانزده': '16', 'هفده': '17', 'هجده': '18', 'نوزده': '19', 'بیست': '20',
        'بیست و یک': '21', 'بیست و دو': '22', 'بیست و سه': '23', 'بیست و چهار': '24',
        'بیست و پنج': '25', 'بیست و شش': '26', 'بیست و هفت': '27', 'بیست و هشت': '28',
        'بیست و نه': '29', 'سی': '30', 'سی و یک': '31', 'سی و دو': '32', 'سی و سه': '33',
        'سی و چهار': '34', 'سی و پنج': '35', 'سی و شش': '36', 'سی و هفت': '37',
        'سی و هشت': '38', 'سی و نه': '39', 'چهل': '40', 'پنجاه': '50', 'شصت': '60',
        'هفتاد': '70', 'هشتاد': '80', 'نود': '90', 'یکصد': '100', 'صد': '100'
    })

    # Persian char mappings - ENHANCED: Better character normalization
    persian_char_mappings: Dict[str, str] = field(default_factory=lambda: {
        'ك': 'ک', 'ي': 'ی', 'ى': 'ی', 'ة': 'ه', 'ۀ': 'ه',
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و', 'ئ': 'ی',
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
        '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
        '\u200c': ' ', '\u200d': '', '\u00a0': ' ',  # Zero-width chars
        'ً': '', 'ٌ': '', 'ٍ': '', 'َ': '', 'ُ': '', 'ِ': '',  # Diacritics
        'ّ': '', 'ْ': '', 'ٰ': '', 'ٱ': 'ا', 'ٲ': 'ا', 'ٳ': 'ا'
    })

    # Casual chat keywords - NEW: Keywords that trigger casual chat mode
    casual_chat_keywords: List[str] = field(default_factory=lambda: [
        'چطوری', 'چطور', 'حالت', 'سلامتی', 'خوبی', 'چه خبر', 'چخبر',
        'احوال', 'کیفیت', 'دوست', 'عزیز', 'جان', 'عشق', 'گل',
        'بابا', 'مامان', 'خانواده', 'زندگی', 'کار', 'شغل',
        'هوا', 'آب و هوا', 'غذا', 'نوشیدنی', 'فیلم', 'موزیک',
        'ورزش', 'فوتبال', 'بازی', 'سرگرمی', 'تفریح', 'مسافرت',
        'عاشق', 'دوست دارم', 'نفرت', 'متنفرم', 'خسته', 'گشنه',
        'تشنه', 'سرد', 'گرم', 'خوشحال', 'غمگین', 'ناراحت'
    ])

    # Lesson titles from the book (accurate names)
    lesson_titles: List[str] = field(default_factory=lambda: [
        "تاریخ‌نگاری و منابع دوره معاصر",
        "ایران و جهان در آستانه دوره معاصر",
        "سیاست و حکومت در عصر قاجار",
        "اوضاع اجتماعی، اقتصادی و فرهنگی عصر قاجار",
        "نهضت مشروطه ایران",
        "جنگ جهانی اول و ایران",
        "ایران در دوره حکومت رضاشاه",
        "جنگ جهانی دوم و جهان پس از آن",
        "نهضت ملی شدن صنعت نفت ایران",
        "انقلاب اسلامی",
        "استقرار و تثبیت نظام جمهوری اسلامی",
        "جنگ تحمیلی و دفاع مقدس"
    ])

    def __post_init__(self):
        """Validate configuration"""
        if not self.openai_api_key or self.openai_api_key == "sk-YOUR_API_KEY_HERE":
            raise ValueError("Valid OpenAI API key required")

        # Log configuration
        logger.info(f"RAG Config v5.1 initialized with primary model: {self.primary_model}")
        logger.info(f"Orchestrator: {'Enabled' if self.enable_orchestrator else 'Disabled'}")
        logger.info(f"Narrative Mode: {'Enabled' if self.enable_narrative_mode else 'Disabled'}")
        logger.info(f"Casual Chat: {'Enabled' if self.enable_casual_chat else 'Disabled'}")  # NEW
        logger.info(f"Universal Intelligence: {'Enabled' if self.enable_universal_intelligence else 'Disabled'}")
        logger.info(f"Advanced Memory: {'Enabled' if self.enable_advanced_memory else 'Disabled'}")
        logger.info(f"Emotional Intelligence: {'Enabled' if self.enable_emotional_intelligence else 'Disabled'}")


# ==================== Logging Configuration ====================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for better readability"""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Configure logging
log_format = '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(log_format))

file_handler = logging.FileHandler('edurag_v51.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)


# ==================== Type Definitions ====================

class QueryType(str, Enum):
    """Enhanced query type enumeration with casual chat support"""
    PAGE_BASED = "page_based"
    ACTIVITY_BASED = "activity_based"
    LESSON_BASED = "lesson_based"
    CONCEPT_SEARCH = "concept_search"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    NAVIGATION = "navigation"
    GREETING = "greeting"
    SUMMARY = "summary"
    CASUAL_CHAT = "casual_chat"  # NEW: For casual conversations
    UNKNOWN = "unknown"


class ResponseStrategy(str, Enum):
    """Response generation strategies"""
    DIRECT_ANSWER = "direct_answer"
    NARRATIVE_EXPLANATION = "narrative_explanation"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    HISTORICAL_CONTEXT = "historical_context"
    PEDAGOGICAL_GUIDANCE = "pedagogical_guidance"
    INTERACTIVE_EXPLORATION = "interactive_exploration"
    ANALYTICAL_EXPLANATION = "analytical_explanation"
    CASUAL_CONVERSATION = "casual_conversation"  # NEW: For casual chat


@dataclass
class QueryContext:
    """Enhanced query context with rich metadata"""
    original_query: str
    normalized_query: str
    query_type: QueryType
    intent: str
    entities: Dict[str, List[Any]]
    complexity: float
    requires_context: bool
    temporal_reference: Optional[str] = None
    comparative_elements: List[str] = field(default_factory=list)
    pedagogical_goal: Optional[str] = None
    user_knowledge_level: str = "intermediate"
    is_casual_chat: bool = False  # NEW: Flag for casual chat detection


@dataclass
class SearchResult:
    """Enhanced search result with scoring and relevance"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    semantic_score: float
    keyword_score: float
    rerank_score: Optional[float] = None
    relevance_explanation: Optional[str] = None
    context_window: Optional[Dict[str, str]] = None  # Previous/next content


@dataclass
class ResponseContext:
    """Context for response generation"""
    query_context: QueryContext
    search_results: List[SearchResult]
    selected_strategy: ResponseStrategy
    narrative_elements: Dict[str, Any]
    cross_references: List[Dict[str, Any]]
    quality_score: float = 0.0
    pedagogical_notes: List[str] = field(default_factory=list)


# ==================== Enhanced State Definitions ====================

class OrchestratorState(TypedDict):
    """Master orchestrator state"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    query_context: QueryContext
    search_results: List[SearchResult]
    response_context: ResponseContext
    final_response: str
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    user_profile: Dict[str, Any]
    conversation_history: List[Dict[str, str]]


# ==================== Utility Functions ====================

def timing_decorator(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.3f}s")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two Persian texts"""
    # Simple Jaccard similarity for now
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


# ==================== Enhanced Persian Text Processing ====================

class AdvancedPersianNormalizer:
    """Advanced Persian text normalization with linguistic understanding and enhanced number recognition"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.char_mappings = config.persian_char_mappings
        self.persian_numbers = config.persian_numbers

        # Compile patterns for efficiency
        self.patterns = {
            'activity': re.compile(r'فعّ?ـ?الی[تّ]?\s*(\d+)', re.IGNORECASE),
            'lesson': re.compile(r'درس\s*(\d+)', re.IGNORECASE),
            'page': re.compile(r'صفحه\s*(\d+)', re.IGNORECASE),
            'section': re.compile(r'بخش\s*(\d+)', re.IGNORECASE),
            'chapter': re.compile(r'فصل\s*(\d+)', re.IGNORECASE),
        }

        # Enhanced patterns for Persian numbers - NEW
        self.persian_number_patterns = {}
        for persian, english in self.persian_numbers.items():
            # Create patterns for different contexts
            self.persian_number_patterns[persian] = {
                'page': re.compile(rf'صفحه\s*{re.escape(persian)}\b', re.IGNORECASE),
                'activity': re.compile(rf'فعّ?ـ?الی[تّ]?\s*{re.escape(persian)}\b', re.IGNORECASE),
                'lesson': re.compile(rf'درس\s*{re.escape(persian)}\b', re.IGNORECASE),
                'general': re.compile(rf'\b{re.escape(persian)}\b', re.IGNORECASE)
            }

        # Common misspellings and corrections
        self.corrections = {
            'مشروطیت': 'مشروطه',
            'قاجاریه': 'قاجار',
            'پهلوی اول': 'رضاشاه',
            'پهلوی دوم': 'محمدرضا پهلوی',
            'انقلاب ۵۷': 'انقلاب اسلامی',
            'جنگ هشت ساله': 'دفاع مقدس',
            'ملی شدن نفت': 'نهضت ملی شدن صنعت نفت',
        }

    def normalize(self, text: str) -> str:
        """Comprehensive text normalization with enhanced Persian number support"""
        if not text:
            return ""

        normalized = text.strip()

        # First, convert Persian numbers to English numbers
        normalized = self._convert_persian_numbers(normalized)

        # Apply character mappings
        for old, new in self.char_mappings.items():
            normalized = normalized.replace(old, new)

        # Apply corrections
        for wrong, correct in self.corrections.items():
            normalized = normalized.replace(wrong, correct)

        # Normalize whitespace and clean up
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = re.sub(r'[‌]+', ' ', normalized)  # Zero-width non-joiner to space

        return normalized.strip()

    def _convert_persian_numbers(self, text: str) -> str:
        """Convert Persian number words to English digits - ENHANCED"""
        result = text

        # Sort by length (longest first) to handle compound numbers properly
        sorted_numbers = sorted(self.persian_numbers.items(), key=lambda x: len(x[0]), reverse=True)

        for persian, english in sorted_numbers:
            # Convert in different contexts
            for context, pattern in self.persian_number_patterns[persian].items():
                if context == 'page':
                    result = pattern.sub(f'صفحه {english}', result)
                elif context == 'activity':
                    result = pattern.sub(f'فعالیت {english}', result)
                elif context == 'lesson':
                    result = pattern.sub(f'درس {english}', result)
                else:
                    # General replacement, but be careful not to replace parts of words
                    result = pattern.sub(english, result)

        return result

    def extract_entities(self, text: str) -> Dict[str, List[Any]]:
        """Extract educational entities from text with improved accuracy"""
        normalized = self.normalize(text)
        entities = {
            'activities': [],
            'lessons': [],
            'pages': [],
            'sections': [],
            'chapters': [],
            'concepts': [],
            'people': [],
            'events': [],
            'dates': []
        }

        # Extract structured entities with improved patterns
        for entity_type, pattern in self.patterns.items():
            matches = pattern.finditer(normalized)
            for match in matches:
                try:
                    number = int(match.group(1))
                    entities[f'{entity_type}s'].append({
                        'number': number,
                        'text': match.group(0),
                        'position': match.start()
                    })
                except ValueError:
                    continue

        # Additional extraction for better accuracy
        # Look for isolated numbers that might be page/lesson references
        isolated_numbers = re.finditer(r'\b(\d+)\b', normalized)
        for match in isolated_numbers:
            number = int(match.group(1))
            position = match.start()

            # Context-based classification
            before_text = normalized[max(0, position - 20):position].lower()
            after_text = normalized[position:position + 20].lower()

            if any(word in before_text or word in after_text for word in ['صفحه', 'ص']):
                if not any(p['number'] == number for p in entities['pages']):
                    entities['pages'].append({
                        'number': number,
                        'text': f'صفحه {number}',
                        'position': position
                    })
            elif any(word in before_text or word in after_text for word in ['درس', 'فصل']):
                if not any(l['number'] == number for l in entities['lessons']):
                    entities['lessons'].append({
                        'number': number,
                        'text': f'درس {number}',
                        'position': position
                    })
            elif any(word in before_text or word in after_text for word in ['فعالیت']):
                if not any(a['number'] == number for a in entities['activities']):
                    entities['activities'].append({
                        'number': number,
                        'text': f'فعالیت {number}',
                        'position': position
                    })

        # Extract historical figures with better patterns
        historical_figures = [
            'رضاشاه', 'رضا شاه', 'محمدرضا شاه', 'محمدرضا پهلوی', 'مصدق', 'دکتر مصدق',
            'امیرکبیر', 'عباس میرزا', 'ناصرالدین شاه', 'مظفرالدین شاه', 'احمد شاه',
            'میرزا تقی خان', 'آقا محمد خان', 'فتحعلی شاه', 'خمینی', 'امام خمینی',
            'هاشمی رفسنجانی', 'خاتمی', 'احمدی‌نژاد', 'روحانی'
        ]

        for figure in historical_figures:
            if figure.lower() in normalized.lower():
                entities['people'].append(figure)

        # Extract historical events with improved recognition
        events = [
            'انقلاب مشروطه', 'نهضت مشروطه', 'کودتای ۱۲۹۹', 'کودتای رضاخان',
            'ملی شدن نفت', 'نهضت ملی شدن صنعت نفت', 'انقلاب اسلامی', 'انقلاب ۵۷',
            'جنگ جهانی اول', 'جنگ جهانی دوم', 'دفاع مقدس', 'جنگ تحمیلی',
            'جنگ ایران و عراق', 'عاشورای خونین', '۱۵ خرداد', 'قیام ۱۵ خرداد'
        ]

        for event in events:
            if any(part.lower() in normalized.lower() for part in event.split()):
                if event not in entities['events']:
                    entities['events'].append(event)

        # Extract dates with better patterns
        date_patterns = [
            r'\b\d{4}\b',  # 4-digit years
            r'\b۱[۲۳][۰-۹]{2}\b',  # Persian 4-digit years
            r'\b\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}\b',  # Date formats
        ]

        for pattern in date_patterns:
            dates = re.findall(pattern, normalized)
            entities['dates'].extend(dates)

        # Remove duplicates while preserving order
        for key in entities:
            if isinstance(entities[key], list):
                seen = set()
                unique_items = []
                for item in entities[key]:
                    if isinstance(item, dict):
                        identifier = (item.get('number'), item.get('text'))
                    else:
                        identifier = item

                    if identifier not in seen:
                        seen.add(identifier)
                        unique_items.append(item)

                entities[key] = unique_items

        return entities


# ==================== Enhanced Query Understanding ====================

class IntelligentQueryAnalyzer:
    """Advanced query analysis with deep understanding and casual chat detection"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.normalizer = AdvancedPersianNormalizer(config)
        self.llm = ChatOpenAI(
            model=config.fast_model,
            temperature=0.1,
            openai_api_key=config.openai_api_key
        )

        # Casual chat detection keywords
        self.casual_keywords = set(config.casual_chat_keywords)

        # Educational keywords for better detection
        self.educational_keywords = {
            'تاریخ', 'درس', 'صفحه', 'فعالیت', 'کتاب', 'مطالب', 'محتوا',
            'توضیح', 'تشریح', 'معنی', 'مفهوم', 'تعریف', 'چیست', 'کیست',
            'چرا', 'چگونه', 'کجا', 'کی', 'چه وقت', 'علت', 'دلیل',
            'مشروطه', 'قاجار', 'پهلوی', 'انقلاب', 'جنگ', 'نفت'
        }

    @timing_decorator
    async def analyze_query(self, query: str, conversation_history: List[Dict[str, str]] = None) -> QueryContext:
        """Perform deep query analysis with casual chat detection"""
        normalized = self.normalizer.normalize(query)
        entities = self.normalizer.extract_entities(query)

        # Check for casual chat first - NEW
        is_casual = self._detect_casual_chat(query, normalized)

        # Build context from conversation history
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            history_context = "\n".join([
                f"کاربر: {h.get('user', '')}\nدستیار: {h.get('assistant', '')[:100]}..."
                for h in recent_history
            ])

        # If it's casual chat, handle differently
        if is_casual:
            return QueryContext(
                original_query=query,
                normalized_query=normalized,
                query_type=QueryType.CASUAL_CHAT,
                intent="گفتگوی دوستانه و غیرآموزشی",
                entities={},  # No educational entities needed
                complexity=0.2,  # Low complexity
                requires_context=False,
                is_casual_chat=True,
                user_knowledge_level="general"
            )

        # Enhanced analysis prompt for non-casual queries
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """شما یک تحلیلگر خبره در درک سوالات آموزشی به زبان فارسی هستید.

وظیفه شما: تحلیل عمیق و دقیق قصد کاربر و ارائه یک تحلیل جامع است.

نکات مهم تحلیل:
1. نوع سوال و قصد کاربر را دقیق تشخیص دهید
2. پیچیدگی و نیاز به زمینه را ارزیابی کنید
3. اهداف آموزشی و سطح دانش کاربر را تعیین کنید
4. ارجاعات زمانی و عناصر مقایسه‌ای را شناسایی کنید
5. سطح ابهام را مشخص کنید
6. دقت ویژه‌ای روی تشخیص صفحات، فعالیت‌ها و درس‌ها داشته باشید

انواع سوالات:
- page_based: سوالات مربوط به صفحه خاص
- activity_based: سوالات مربوط به فعالیت خاص
- lesson_based: سوالات مربوط به درس خاص
- concept_search: جستجوی مفاهیم و توضیحات
- comparative: مقایسه بین موضوعات
- analytical: تحلیل عمیق موضوعات
- greeting: سلام و احوال‌پرسی
- summary: درخواست خلاصه

قوانین مطلق:
- فقط و فقط یک JSON خروجی دهید
- هیچ توضیح اضافی قبل یا بعد از JSON ندهید
- همه مقادیر باید به فارسی باشند (به جز نام فیلدها)
- حتماً همه فیلدها را پر کنید

ساختار دقیق JSON خروجی:
{{
    "query_type": "یکی از: page_based, activity_based, lesson_based, concept_search, comparative, analytical, navigation, greeting, summary",
    "intent": "توصیف واضح و کامل از آنچه کاربر می‌خواهد",
    "complexity": عدد بین 0.0 تا 1.0,
    "requires_context": true یا false,
    "temporal_reference": "past یا present یا future یا none",
    "comparative_elements": ["عنصر اول", "عنصر دوم"],
    "pedagogical_goal": "هدف یادگیری در صورت وجود",
    "suggested_knowledge_level": "beginner یا intermediate یا advanced",
    "key_concepts": ["مفهوم۱", "مفهوم۲"],
    "ambiguity_level": "none یا low یا medium یا high",
    "clarification_needed": "توضیح خاص در صورت ابهام بالا"
}}"""),
            ("human", """سوال کاربر: {query}
نرمال‌سازی شده: {normalized}
موجودیت‌های استخراج شده: {entities}
تاریخچه اخیر: {history_context}

تحلیل کنید و فقط JSON را برگردانید:""")
        ])

        try:
            # Get LLM analysis
            chain = analysis_prompt | self.llm | JsonOutputParser()
            analysis = await chain.ainvoke({
                "query": query,
                "normalized": normalized,
                "entities": json.dumps(entities, ensure_ascii=False),
                "history_context": history_context
            })

            # Determine query type with improved accuracy
            query_type = QueryType(analysis.get("query_type", "unknown"))

            # Override with pattern matching for certain types - IMPROVED
            if entities.get('pages') and not entities.get('activities'):
                query_type = QueryType.PAGE_BASED
            elif entities.get('activities'):
                query_type = QueryType.ACTIVITY_BASED
            elif entities.get('lessons') and not entities.get('pages') and not entities.get('activities'):
                query_type = QueryType.LESSON_BASED
            elif any(greeting in query.lower() for greeting in ["سلام", "درود", "صبح بخیر", "عصر بخیر"]):
                query_type = QueryType.GREETING
            elif any(summary_word in query.lower() for summary_word in ["خلاصه", "چکیده", "مختصر"]):
                query_type = QueryType.SUMMARY
            elif any(compare_word in query.lower() for compare_word in ["مقایسه", "تفاوت", "شباهت", "برابر"]):
                query_type = QueryType.COMPARATIVE

            return QueryContext(
                original_query=query,
                normalized_query=normalized,
                query_type=query_type,
                intent=analysis.get("intent", "نامشخص"),
                entities=entities,
                complexity=float(analysis.get("complexity", 0.5)),
                requires_context=analysis.get("requires_context", False),
                temporal_reference=analysis.get("temporal_reference"),
                comparative_elements=analysis.get("comparative_elements", []),
                pedagogical_goal=analysis.get("pedagogical_goal"),
                user_knowledge_level=analysis.get("suggested_knowledge_level", "intermediate"),
                is_casual_chat=False
            )

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Fallback to basic analysis
            return QueryContext(
                original_query=query,
                normalized_query=normalized,
                query_type=self._determine_basic_type(query, entities),
                intent="جستجوی اطلاعات",
                entities=entities,
                complexity=0.5,
                requires_context=False,
                is_casual_chat=is_casual
            )

    def _detect_casual_chat(self, original_query: str, normalized_query: str) -> bool:
        """Detect if the query is casual chat - NEW"""
        query_lower = original_query.lower()
        normalized_lower = normalized_query.lower()

        # Check for casual keywords
        casual_score = 0
        educational_score = 0

        # Count casual indicators
        for keyword in self.casual_keywords:
            if keyword in query_lower or keyword in normalized_lower:
                casual_score += 1

        # Count educational indicators
        for keyword in self.educational_keywords:
            if keyword in query_lower or keyword in normalized_lower:
                educational_score += 1

        # Additional casual patterns
        casual_patterns = [
            r'چطور(ی)?',
            r'چه خبر',
            r'حالت چطوره',
            r'خوبی',
            r'چی کار می‌?کنی',
            r'دوست دارم',
            r'عاشقتم',
            r'خسته‌?ام',
            r'گشنه‌?ام',
            r'چه فیلمی',
            r'چه موزیکی',
        ]

        for pattern in casual_patterns:
            if re.search(pattern, query_lower):
                casual_score += 2

        # Simple questions that might be educational but phrased casually
        simple_educational = [
            r'این چیه',
            r'اون چیه',
            r'یعنی چی',
            r'معنیش چیه',
        ]

        for pattern in simple_educational:
            if re.search(pattern, query_lower):
                educational_score += 1

        # Decision logic
        # If educational score is high, it's probably educational
        if educational_score >= 2:
            return False

        # If casual score is higher or equal and no clear educational content
        if casual_score >= 1 and educational_score == 0:
            return True

        # Check query length - very short queries are often casual
        if len(original_query.split()) <= 2 and casual_score > 0:
            return True

        return False

    def _determine_basic_type(self, query: str, entities: Dict[str, List[Any]]) -> QueryType:
        """Basic query type determination with improved accuracy"""
        if entities.get('pages'):
            return QueryType.PAGE_BASED
        elif entities.get('activities'):
            return QueryType.ACTIVITY_BASED
        elif entities.get('lessons'):
            return QueryType.LESSON_BASED
        elif "سلام" in query.lower() or "درود" in query.lower():
            return QueryType.GREETING
        elif "خلاصه" in query.lower():
            return QueryType.SUMMARY
        elif any(word in query.lower() for word in ["مقایسه", "تفاوت", "شباهت"]):
            return QueryType.COMPARATIVE
        else:
            return QueryType.CONCEPT_SEARCH


# ==================== Advanced Memory System ====================

class GraphBasedMemory:
    """Graph-based memory system for relationship tracking"""

    def __init__(self):
        self.user_profiles = {}
        self.concept_graph = defaultdict(lambda: defaultdict(float))
        self.interaction_history = defaultdict(list)
        self.learning_paths = defaultdict(list)

    def update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update user profile with new interaction"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "created_at": datetime.now(),
                "knowledge_level": "intermediate",
                "interests": [],
                "completed_topics": set(),
                "learning_style": "balanced",
                "interaction_count": 0,
                "preferred_depth": "medium",
                "success_rate": 1.0,
                "casual_chat_count": 0  # NEW: Track casual conversations
            }

        profile = self.user_profiles[user_id]
        profile["interaction_count"] += 1
        profile["last_interaction"] = datetime.now()

        # NEW: Track casual chat usage
        if interaction_data.get("is_casual_chat"):
            profile["casual_chat_count"] += 1

        # Update interests
        if "concepts" in interaction_data:
            for concept in interaction_data["concepts"]:
                if concept not in profile["interests"]:
                    profile["interests"].append(concept)

        # Update completed topics
        if interaction_data.get("topic_completed"):
            profile["completed_topics"].add(interaction_data["topic_completed"])

        # Adjust knowledge level based on query complexity
        if "complexity" in interaction_data:
            self._adjust_knowledge_level(profile, interaction_data["complexity"])

    def _adjust_knowledge_level(self, profile: Dict[str, Any], complexity: float):
        """Dynamically adjust user's knowledge level"""
        current_level = profile["knowledge_level"]

        if complexity > 0.8 and profile["success_rate"] > 0.7:
            if current_level == "intermediate":
                profile["knowledge_level"] = "advanced"
        elif complexity < 0.3 and current_level == "advanced":
            profile["knowledge_level"] = "intermediate"
        elif complexity < 0.2 and current_level == "intermediate":
            profile["knowledge_level"] = "beginner"

    def track_concept_relationship(self, concept1: str, concept2: str, strength: float = 1.0):
        """Track relationships between concepts"""
        self.concept_graph[concept1][concept2] += strength
        self.concept_graph[concept2][concept1] += strength  # Bidirectional

    def get_related_concepts(self, concept: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Get concepts related to a given concept"""
        if concept not in self.concept_graph:
            return []

        related = [
            (related_concept, strength)
            for related_concept, strength in self.concept_graph[concept].items()
            if strength >= threshold
        ]

        return sorted(related, key=lambda x: x[1], reverse=True)

    def add_interaction(self, user_id: str, query: str, response: str, metadata: Dict[str, Any]):
        """Add interaction to history"""
        interaction = {
            "timestamp": datetime.now(),
            "query": query,
            "response": response[:200],  # Store summary
            "metadata": metadata
        }

        self.interaction_history[user_id].append(interaction)

        # Keep only recent history
        if len(self.interaction_history[user_id]) > 50:
            self.interaction_history[user_id] = self.interaction_history[user_id][-50:]

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context"""
        profile = self.user_profiles.get(user_id, {})
        recent_interactions = self.interaction_history.get(user_id, [])[-5:]

        # Extract recent topics
        recent_topics = []
        for interaction in recent_interactions:
            if "topics" in interaction.get("metadata", {}):
                recent_topics.extend(interaction["metadata"]["topics"])

        return {
            "profile": profile,
            "recent_topics": list(set(recent_topics)),
            "interaction_patterns": self._analyze_interaction_patterns(user_id),
            "learning_trajectory": self._get_learning_trajectory(user_id)
        }

    def _analyze_interaction_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's interaction patterns"""
        interactions = self.interaction_history.get(user_id, [])

        if not interactions:
            return {"pattern": "new_user"}

        # Analyze query types
        query_types = [i.get("metadata", {}).get("query_type") for i in interactions]
        type_distribution = Counter(query_types)

        # Analyze time patterns
        timestamps = [i["timestamp"] for i in interactions]
        if len(timestamps) > 1:
            time_deltas = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            avg_time_between = sum(time_deltas) / len(time_deltas)
        else:
            avg_time_between = 0

        return {
            "query_type_distribution": dict(type_distribution),
            "avg_time_between_queries": avg_time_between,
            "total_interactions": len(interactions)
        }

    def _get_learning_trajectory(self, user_id: str) -> List[str]:
        """Get user's learning trajectory"""
        return self.learning_paths.get(user_id, [])


# ==================== Enhanced Search System ====================

class ContextAwareSearchEngine:
    """Advanced search engine with context understanding"""

    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.memory = GraphBasedMemory()

        # Initialize reranker
        try:
            if hasattr(config, 'reranker_model'):
                self.reranker = CrossEncoder(config.reranker_model)
            else:
                self.reranker = None
                logger.warning("Reranker model not specified in config")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            self.reranker = None

    @timing_decorator
    async def search(self, query_context: QueryContext, user_context: Dict[str, Any]) -> List[SearchResult]:
        """Perform context-aware search"""
        try:
            # Skip search for casual chat - NEW
            if query_context.is_casual_chat:
                return []

            # Choose search strategy based on query type
            if query_context.query_type == QueryType.PAGE_BASED:
                results = await self._search_by_page(query_context)
            elif query_context.query_type == QueryType.ACTIVITY_BASED:
                results = await self._search_by_activity(query_context)
            elif query_context.query_type == QueryType.LESSON_BASED:
                results = await self._search_by_lesson(query_context)
            elif query_context.query_type == QueryType.COMPARATIVE:
                results = await self._search_comparative(query_context)
            else:
                results = await self._semantic_search(query_context)

            # Enhance results with context
            enhanced_results = await self._enhance_with_context(results, query_context, user_context)

            # Apply relevance filtering
            filtered_results = [r for r in enhanced_results if r.relevance_score >= self.config.min_relevance_score]

            # If too few results, lower threshold and try again
            if len(filtered_results) < 3:
                filtered_results = [r for r in enhanced_results if r.relevance_score >= 0.5]

            return filtered_results[:self.config.final_top_k]

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return []

    async def _search_by_page(self, query_context: QueryContext) -> List[SearchResult]:
        """Search for specific page content"""
        page_numbers = [p['number'] for p in query_context.entities.get('pages', [])]
        results = []

        for page_num in page_numbers:
            chunks = self.vector_store.get_chunks_by_page(page_num)

            if chunks:
                # Aggregate chunks into complete page
                aggregated = self._aggregate_page_chunks(chunks)

                # Get context window (previous and next pages)
                context_window = await self._get_page_context_window(page_num)

                result = SearchResult(
                    chunk_id=f"page_{page_num}",
                    content=aggregated['content'],
                    metadata={
                        'page_number': page_num,
                        'type': 'full_page',
                        'chunk_count': len(chunks),
                        **aggregated['metadata']
                    },
                    relevance_score=1.0,
                    semantic_score=1.0,
                    keyword_score=1.0,
                    relevance_explanation="صفحه درخواستی",
                    context_window=context_window
                )
                results.append(result)

        return results

    async def _search_by_activity(self, query_context: QueryContext) -> List[SearchResult]:
        """Search for activity content"""
        activities = query_context.entities.get('activities', [])
        lessons = query_context.entities.get('lessons', [])
        results = []

        for activity in activities:
            activity_num = activity['number']

            if lessons:
                # Search in specific lessons
                for lesson in lessons:
                    chunks = self.vector_store.get_activity_chunks(activity_num, lesson['number'])
                    if chunks:
                        aggregated = self._aggregate_activity_chunks(chunks)
                        result = SearchResult(
                            chunk_id=f"activity_{activity_num}_lesson_{lesson['number']}",
                            content=aggregated['content'],
                            metadata={
                                'activity_number': activity_num,
                                'lesson_number': lesson['number'],
                                'type': 'full_activity',
                                **aggregated['metadata']
                            },
                            relevance_score=1.0,
                            semantic_score=1.0,
                            keyword_score=1.0,
                            relevance_explanation=f"فعالیت {activity_num} درس {lesson['number']}"
                        )
                        results.append(result)
            else:
                # Search across all lessons
                all_activities = self.vector_store.get_all_activities(activity_num)
                for lesson_num, chunks in all_activities.items():
                    aggregated = self._aggregate_activity_chunks(chunks)
                    result = SearchResult(
                        chunk_id=f"activity_{activity_num}_lesson_{lesson_num}",
                        content=aggregated['content'],
                        metadata={
                            'activity_number': activity_num,
                            'lesson_number': lesson_num,
                            'type': 'full_activity',
                            **aggregated['metadata']
                        },
                        relevance_score=0.9,
                        semantic_score=0.9,
                        keyword_score=1.0,
                        relevance_explanation=f"فعالیت {activity_num} در درس {lesson_num}"
                    )
                    results.append(result)

        return results

    async def _search_by_lesson(self, query_context: QueryContext) -> List[SearchResult]:
        """Search for lesson overview"""
        lessons = query_context.entities.get('lessons', [])
        results = []

        for lesson in lessons:
            lesson_num = lesson['number']
            chunks = self.vector_store.get_lesson_chunks(lesson_num)

            if chunks:
                overview = self._create_lesson_overview(chunks, lesson_num)
                result = SearchResult(
                    chunk_id=f"lesson_{lesson_num}",
                    content=overview['content'],
                    metadata={
                        'lesson_number': lesson_num,
                        'lesson_title': self._get_lesson_title(lesson_num),
                        'type': 'lesson_overview',
                        **overview['metadata']
                    },
                    relevance_score=1.0,
                    semantic_score=1.0,
                    keyword_score=1.0,
                    relevance_explanation=f"محتوای درس {lesson_num}"
                )
                results.append(result)

        return results

    async def _search_comparative(self, query_context: QueryContext) -> List[SearchResult]:
        """Search for comparative analysis"""
        elements = query_context.comparative_elements
        results = []

        # Search for each element
        for element in elements:
            element_results = await self._semantic_search_for_concept(element)
            results.extend(element_results)

        # Find connections between elements
        if len(elements) >= 2:
            connection_results = await self._find_conceptual_connections(elements)
            results.extend(connection_results)

        return results

    async def _semantic_search(self, query_context: QueryContext) -> List[SearchResult]:
        """Perform semantic search with entity boosting"""
        query = query_context.normalized_query

        # Get base semantic results
        raw_results = await self.vector_store.semantic_search(query, k=self.config.retrieval_top_k)

        results = []
        for raw in raw_results:
            # Calculate relevance with entity boosting
            relevance_score = self._calculate_relevance_score(raw, query_context)

            # Create SearchResult
            result = SearchResult(
                chunk_id=raw.get('chunk_id', ''),
                content=raw['content'],
                metadata=raw.get('metadata', {}),
                relevance_score=relevance_score,
                semantic_score=raw.get('score', 0.0),
                keyword_score=raw.get('keyword_score', 0.0)
            )
            results.append(result)

        # Rerank if available
        if self.reranker and len(results) > self.config.rerank_top_k:
            results = await self._rerank_results(query, results[:self.config.rerank_top_k])

        return results

    async def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        if not self.reranker:
            return results

        try:
            pairs = [[query, r.content] for r in results]
            scores = self.reranker.predict(pairs)

            for i, score in enumerate(scores):
                results[i].rerank_score = float(score)
                # Update relevance score with rerank score
                results[i].relevance_score = (
                        0.4 * results[i].relevance_score +
                        0.6 * results[i].rerank_score
                )

            results.sort(key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            logger.error(f"Reranking error: {e}")

        return results

    async def _enhance_with_context(self, results: List[SearchResult],
                                    query_context: QueryContext,
                                    user_context: Dict[str, Any]) -> List[SearchResult]:
        """Enhance results with contextual information"""
        for result in results:
            # Add relevance explanation
            result.relevance_explanation = await self._generate_relevance_explanation(
                result, query_context
            )

            # Add difficulty assessment
            result.metadata['difficulty_level'] = self._assess_difficulty(
                result.content, user_context.get('profile', {})
            )

            # Add pedagogical metadata
            result.metadata['pedagogical_value'] = self._assess_pedagogical_value(
                result, query_context
            )

            # Track concept relationships
            concepts = result.metadata.get('keywords', [])
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    self.memory.track_concept_relationship(concepts[i], concepts[j])

        return results

    async def _generate_relevance_explanation(self, result: SearchResult,
                                              query_context: QueryContext) -> str:
        """Generate explanation for why result is relevant"""
        explanations = []

        # Check entity matches
        for entity_type, entities in query_context.entities.items():
            if entities and entity_type in ['pages', 'activities', 'lessons']:
                for entity in entities:
                    if str(entity.get('number', '')) in str(
                            result.metadata.get(entity_type.rstrip('s') + '_number', '')):
                        explanations.append(f"مطابقت مستقیم با {entity_type}")

        # Check keyword matches
        query_keywords = set(query_context.normalized_query.lower().split())
        content_keywords = set(result.content.lower().split())
        common_keywords = query_keywords.intersection(content_keywords)

        if len(common_keywords) > 3:
            explanations.append(f"تطابق کلیدواژه‌ها: {len(common_keywords)} مورد")

        # Check semantic relevance
        if result.semantic_score > 0.8:
            explanations.append("ارتباط معنایی قوی")
        elif result.semantic_score > 0.6:
            explanations.append("ارتباط معنایی متوسط")

        return " | ".join(explanations) if explanations else "ارتباط کلی"

    def _calculate_relevance_score(self, raw_result: Dict[str, Any],
                                   query_context: QueryContext) -> float:
        """Calculate comprehensive relevance score"""
        base_score = raw_result.get('score', 0.5)

        # Entity boost
        entity_boost = 0.0
        content_lower = raw_result['content'].lower()

        for entity_type, entities in query_context.entities.items():
            if entities:
                for entity in entities:
                    if isinstance(entity, dict):
                        if str(entity.get('text', '')).lower() in content_lower:
                            entity_boost += 0.1
                    elif str(entity).lower() in content_lower:
                        entity_boost += 0.1

        # Recency boost (if metadata available)
        recency_boost = 0.0
        if query_context.temporal_reference == "present":
            # Boost recent content
            pass

        # User preference boost
        preference_boost = 0.0

        # Calculate final score
        final_score = min(1.0, base_score + entity_boost + recency_boost + preference_boost)

        return final_score

    def _assess_difficulty(self, content: str, user_profile: Dict[str, Any]) -> str:
        """Assess content difficulty relative to user"""
        word_count = len(content.split())
        complex_words = len([w for w in content.split() if len(w) > 10])

        # Simple heuristic
        if word_count < 100 and complex_words < 5:
            difficulty = "easy"
        elif word_count < 300 and complex_words < 15:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Adjust based on user level
        user_level = user_profile.get('knowledge_level', 'intermediate')

        if user_level == "beginner" and difficulty == "hard":
            return "very_hard"
        elif user_level == "advanced" and difficulty == "easy":
            return "very_easy"

        return difficulty

    def _assess_pedagogical_value(self, result: SearchResult,
                                  query_context: QueryContext) -> float:
        """Assess pedagogical value of content"""
        value = 0.5  # Base value

        # Check if it matches pedagogical goal
        if query_context.pedagogical_goal:
            # Simple keyword matching for now
            if any(keyword in result.content for keyword in query_context.pedagogical_goal.split()):
                value += 0.2

        # Check content type
        content_type = result.metadata.get('section_type', '')
        if content_type == 'activity':
            value += 0.2  # Activities have high pedagogical value
        elif content_type == 'example':
            value += 0.15
        elif content_type == 'summary':
            value += 0.1

        return min(1.0, value)

    async def _get_page_context_window(self, page_num: int) -> Dict[str, str]:
        """Get context from surrounding pages"""
        context = {}

        # Get previous page summary
        if page_num > 1:
            prev_chunks = self.vector_store.get_chunks_by_page(page_num - 1)
            if prev_chunks:
                context['previous'] = self._summarize_chunks(prev_chunks)

        # Get next page summary
        if page_num < 200:  # Assuming max 200 pages
            next_chunks = self.vector_store.get_chunks_by_page(page_num + 1)
            if next_chunks:
                context['next'] = self._summarize_chunks(next_chunks)

        return context

    async def _semantic_search_for_concept(self, concept: str) -> List[SearchResult]:
        """Search for a specific concept"""
        results = await self.vector_store.semantic_search(concept, k=5)

        search_results = []
        for raw in results:
            result = SearchResult(
                chunk_id=raw.get('chunk_id', ''),
                content=raw['content'],
                metadata=raw.get('metadata', {}),
                relevance_score=raw.get('score', 0.0),
                semantic_score=raw.get('score', 0.0),
                keyword_score=raw.get('keyword_score', 0.0),
                relevance_explanation=f"محتوای مرتبط با {concept}"
            )
            search_results.append(result)

        return search_results

    async def _find_conceptual_connections(self, concepts: List[str]) -> List[SearchResult]:
        """Find content that connects multiple concepts"""
        # Search for content containing multiple concepts
        combined_query = " ".join(concepts)
        results = await self.vector_store.semantic_search(combined_query, k=10)

        connection_results = []
        for raw in results:
            # Check how many concepts are present
            content_lower = raw['content'].lower()
            concept_count = sum(1 for concept in concepts if concept.lower() in content_lower)

            if concept_count >= 2:  # At least 2 concepts present
                result = SearchResult(
                    chunk_id=raw.get('chunk_id', ''),
                    content=raw['content'],
                    metadata=raw.get('metadata', {}),
                    relevance_score=raw.get('score', 0.0) * (concept_count / len(concepts)),
                    semantic_score=raw.get('score', 0.0),
                    keyword_score=raw.get('keyword_score', 0.0),
                    relevance_explanation=f"ارتباط بین {' و '.join(concepts)}"
                )
                connection_results.append(result)

        return connection_results

    def _aggregate_page_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate chunks from the same page"""
        # Sort by position
        chunks.sort(key=lambda x: x.get('metadata', {}).get('position', 0))

        # Combine content
        combined_content = "\n\n".join([c.get('content', '') for c in chunks])

        # Merge metadata
        all_keywords = []
        for chunk in chunks:
            all_keywords.extend(chunk.get('keywords', []))

        unique_keywords = list(set(all_keywords))

        return {
            'content': combined_content,
            'metadata': {
                'keywords': unique_keywords,
                'chunk_count': len(chunks)
            }
        }

    def _aggregate_activity_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate chunks for an activity"""
        return self._aggregate_page_chunks(chunks)  # Same logic for now

    def _create_lesson_overview(self, chunks: List[Dict[str, Any]], lesson_num: int) -> Dict[str, Any]:
        """Create comprehensive lesson overview"""
        # Group chunks by type
        sections = defaultdict(list)
        for chunk in chunks:
            section_type = chunk.get('metadata', {}).get('section_type', 'content')
            sections[section_type].append(chunk)

        # Build overview
        overview_parts = []

        # Title - استفاده از عنوان صحیح
        lesson_title = self._get_lesson_title(lesson_num)
        overview_parts.append(f"# درس {lesson_num}: {lesson_title}\n")

        # محتوای اصلی از خود چانک‌ها
        if chunks:
            # استفاده از محتوای واقعی چانک‌ها
            overview_parts.append("## نکات کلیدی درس:")

            # جمع‌آوری محتوای مهم
            important_content = []
            for chunk in chunks[:5]:  # بررسی 5 چانک اول
                content = chunk.get('content', '')
                if content and len(content) > 50:
                    # استخراج جملات مهم
                    sentences = content.split('.')
                    for sentence in sentences[:2]:
                        if len(sentence.strip()) > 20:
                            important_content.append(sentence.strip())

            # نمایش محتوای مهم
            for i, content in enumerate(important_content[:5], 1):
                overview_parts.append(f"{i}. {content}")

        # Activities
        activity_chunks = [c for c in chunks if 'فعالیت' in c.get('content', '')]
        if activity_chunks:
            overview_parts.append("\n## فعالیت‌های درس:")
            activities_found = set()
            for chunk in activity_chunks:
                activity_match = re.search(r'فعالیت\s*(\d+)', chunk['content'])
                if activity_match:
                    activities_found.add(int(activity_match.group(1)))

            for activity_num in sorted(activities_found):
                overview_parts.append(f"• فعالیت {activity_num}")

        # Keywords from actual content
        all_keywords = []
        for chunk in chunks:
            keywords = chunk.get('keywords', [])
            all_keywords.extend(keywords)

        # شمارش تکرار کلیدواژه‌ها
        keyword_counts = Counter(all_keywords)
        top_keywords = [k for k, v in keyword_counts.most_common(10)]

        if top_keywords:
            overview_parts.append(f"\n## کلیدواژه‌های مهم: {', '.join(top_keywords)}")

        return {
            'content': "\n".join(overview_parts),
            'metadata': {
                'lesson_title': lesson_title,
                'sections': list(sections.keys()),
                'total_chunks': len(chunks),
                'keywords': top_keywords,
                'has_activities': len(activity_chunks) > 0
            }
        }

    def _get_lesson_title(self, lesson_num: int) -> str:
        """Get lesson title from config"""
        lesson_titles = [
            "تاریخ‌نگاری و منابع دوره معاصر",  # درس 1
            "ایران و جهان در آستانه دوره معاصر",  # درس 2
            "سیاست و حکومت در عصر قاجار",  # درس 3
            "اوضاع اجتماعی، اقتصادی و فرهنگی عصر قاجار",  # درس 4
            "نهضت مشروطه ایران",  # درس 5
            "جنگ جهانی اول و ایران",  # درس 6
            "ایران در دوره حکومت رضاشاه",  # درس 7
            "جنگ جهانی دوم و جهان پس از آن",  # درس 8
            "نهضت ملی شدن صنعت نفت ایران",  # درس 9
            "انقلاب اسلامی",  # درس 10
            "استقرار و تثبیت نظام جمهوری اسلامی",  # درس 11
            "جنگ تحمیلی و دفاع مقدس"  # درس 12
        ]

        if 1 <= lesson_num <= len(lesson_titles):
            return lesson_titles[lesson_num - 1]
        return f"درس {lesson_num}"

    def _summarize_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Create brief summary of chunks"""
        if not chunks:
            return ""

        # Take first 200 characters from first few chunks
        summaries = []
        for chunk in chunks[:2]:
            content = chunk.get('content', '')
            if content:
                summary = content[:150] + "..."
                summaries.append(summary)

        return " | ".join(summaries)


# ==================== Enhanced Master Orchestrator ====================

class MasterOrchestrator:
    """Master orchestrator for intelligent response coordination with casual chat support"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.primary_model,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key
        )

        # Response strategy selector
        self.strategy_selector_prompt = ChatPromptTemplate.from_messages([
            ("system", """شما یک هماهنگ‌کننده آموزشی حرفه‌ای هستید که بهترین استراتژی پاسخ‌دهی را انتخاب می‌کنید.

استراتژی‌های موجود:
1. direct_answer: پاسخ ساده و مستقیم برای سوالات واضح
2. narrative_explanation: توضیح داستان‌گونه با زمینه‌سازی کامل
3. comparative_analysis: مقایسه و تحلیل تفاوت‌ها و شباهت‌ها
4. historical_context: قرار دادن در چارچوب تاریخی
5. pedagogical_guidance: راهنمایی گام‌به‌گام آموزشی
6. interactive_exploration: تشویق به کشف از طریق پرسش‌های هدایت‌شده
7. casual_conversation: گفتگوی دوستانه و غیرآموزشی

معیارهای انتخاب:
- نوع سوال و میزان پیچیدگی
- سطح دانش کاربر
- اهداف آموزشی
- کیفیت محتوای موجود
- نیاز به تعامل و درگیری ذهنی
- آیا سوال آموزشی است یا گفتگوی عادی

قانون مطلق: فقط یک JSON با فرمت زیر برگردانید، بدون هیچ توضیح اضافی:
{{"strategy": "نام_استراتژی_انتخابی", "reasoning": "دلیل انتخاب این استراتژی"}}"""),
            ("human", """سوال: {query}
نوع سوال: {query_type}
پیچیدگی: {complexity}
سطح کاربر: {user_level}
کیفیت نتایج جستجو: {results_quality}
هدف آموزشی: {pedagogical_goal}
آیا گفتگوی عادی است: {is_casual}

بهترین استراتژی را انتخاب کنید:""")
        ])

        # Casual chat response prompt - NEW
        self.casual_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """شما یک دستیار دوستانه و صمیمی هستید که با کاربران به صورت طبیعی و گرم گفتگو می‌کنید.

ویژگی‌های شما:
- دوستانه و صمیمی
- طبیعی و انسانی
- مثبت و انرژی‌بخش
- قابل اعتماد و یاری‌رسان
- احترام‌آمیز و مودب

قوانین گفتگو:
1. به صورت طبیعی و دوستانه پاسخ دهید
2. از زبان محاوره‌ای و صمیمی استفاده کنید
3. اگر سوال شخصی پرسیدند، محدودیت‌هایتان را به شکل دوستانه توضیح دهید
4. سعی کنید گفتگو را جذاب و مثبت نگه دارید
5. اگر کاربر نیاز به کمک آموزشی داشت، او را به موضوعات تاریخی راهنمایی کنید

نکته: شما یک دستیار آموزشی تاریخ هستید که قابلیت گفتگوی دوستانه هم دارید."""),
            ("human", "{query}")
        ])

    async def orchestrate_response(self, state: OrchestratorState) -> Dict[str, Any]:
        """Orchestrate the entire response generation process"""
        try:
            # Check if it's casual chat - NEW
            if state["query_context"].is_casual_chat:
                return await self._handle_casual_chat(state)

            # Continue with regular educational flow
            # 1. Select response strategy
            strategy = await self._select_response_strategy(state)
            logger.info(f"Selected strategy: {strategy}")

            # 2. Prepare response context
            response_context = await self._prepare_response_context(state, strategy)

            # 3. Generate initial response
            initial_response = await self._generate_initial_response(response_context)

            # 4. Enhance response with context
            enhanced_response = await self._enhance_response(initial_response, response_context)

            # 5. Quality check
            quality_score = await self._assess_response_quality(enhanced_response, state)

            # 6. Refine if needed
            if quality_score < 0.7:
                enhanced_response = await self._refine_response(enhanced_response, response_context)
                quality_score = min(0.8, quality_score + 0.1)  # Slight boost after refinement

            # 7. Add pedagogical elements
            final_response = await self._add_pedagogical_elements(enhanced_response, response_context)

            # 8. Generate metadata
            metadata = self._generate_response_metadata(response_context, quality_score)

            return {
                "response": final_response,
                "metadata": metadata,
                "quality_score": quality_score,
                "strategy_used": strategy.value
            }

        except Exception as e:
            logger.error(f"Orchestration error: {e}", exc_info=True)
            return {
                "response": "متأسفانه در پردازش درخواست شما مشکلی پیش آمد. لطفاً دوباره تلاش کنید.",
                "metadata": {"error": str(e)},
                "quality_score": 0.0
            }

    async def _handle_casual_chat(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle casual chat conversations - NEW"""
        try:
            query = state["query"]

            # Generate casual response
            chain = self.casual_chat_prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({"query": query})

            # Simple quality assessment for casual chat
            quality_score = 0.8 if len(response) > 10 else 0.5

            # Add gentle educational encouragement occasionally
            if any(word in query.lower() for word in ['کمک', 'یاد', 'بگو', 'چی', 'چطور']):
                response += "\n\n💡 راستی، اگر سوالی از تاریخ معاصر ایران داری، خوشحال می‌شم کمکت کنم!"

            metadata = {
                "response_type": "casual_chat",
                "strategy_used": "casual_conversation",
                "educational_hint": "کمک در مطالب تاریخی نیز ارائه می‌شود"
            }

            return {
                "response": response,
                "metadata": metadata,
                "quality_score": quality_score,
                "strategy_used": "casual_conversation"
            }

        except Exception as e:
            logger.error(f"Casual chat error: {e}")
            return {
                "response": "سلام! چطوری؟ خوشحالم که باهام صحبت می‌کنی! 😊",
                "metadata": {"error": str(e), "response_type": "casual_chat"},
                "quality_score": 0.6
            }

    async def _select_response_strategy(self, state: OrchestratorState) -> ResponseStrategy:
        """Select optimal response strategy"""
        query_context = state["query_context"]
        search_results = state["search_results"]

        # Handle casual chat first
        if query_context.is_casual_chat:
            return ResponseStrategy.CASUAL_CONVERSATION

        # محاسبه کیفیت نتایج
        if search_results:
            avg_relevance = sum(r.relevance_score for r in search_results[:3]) / min(3, len(search_results))
            results_quality = "high" if avg_relevance > 0.8 else "medium" if avg_relevance > 0.5 else "low"
        else:
            results_quality = "none"

        # انتخاب استراتژی بر اساس نوع سوال
        if query_context.query_type == QueryType.GREETING:
            return ResponseStrategy.DIRECT_ANSWER

        if query_context.query_type == QueryType.COMPARATIVE:
            return ResponseStrategy.COMPARATIVE_ANALYSIS

        if query_context.query_type == QueryType.ANALYTICAL:
            return ResponseStrategy.ANALYTICAL_EXPLANATION

        # برای سوالات پیچیده
        if query_context.complexity > 0.7:
            if "چرا" in query_context.original_query or "علت" in query_context.original_query:
                return ResponseStrategy.HISTORICAL_CONTEXT
            else:
                return ResponseStrategy.NARRATIVE_EXPLANATION

        # برای سوالات آموزشی
        if query_context.pedagogical_goal or "توضیح" in query_context.original_query:
            return ResponseStrategy.PEDAGOGICAL_GUIDANCE

        # اگر کاربر خلاصه می‌خواهد
        if "خلاصه" in query_context.original_query or "summary" in query_context.query_type.value:
            return ResponseStrategy.NARRATIVE_EXPLANATION

        try:
            # استفاده از LLM برای تصمیم‌گیری پیچیده‌تر
            chain = self.strategy_selector_prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({
                "query": query_context.original_query,
                "query_type": query_context.query_type.value,
                "complexity": query_context.complexity,
                "user_level": query_context.user_knowledge_level,
                "results_quality": results_quality,
                "pedagogical_goal": query_context.pedagogical_goal or "فهم عمومی",
                "is_casual": query_context.is_casual_chat
            })

            strategy_name = result.get("strategy", "direct_answer")
            logger.info(f"LLM selected strategy: {strategy_name}, reason: {result.get('reasoning', 'N/A')}")

            # اعتبارسنجی استراتژی
            valid_strategies = [s.value for s in ResponseStrategy]
            if strategy_name in valid_strategies:
                return ResponseStrategy(strategy_name)
            else:
                logger.warning(f"Invalid strategy from LLM: {strategy_name}")
                return ResponseStrategy.DIRECT_ANSWER

        except Exception as e:
            logger.warning(f"Strategy selection error: {e}")
            # Fallback strategy
            return ResponseStrategy.DIRECT_ANSWER

    async def _prepare_response_context(self, state: OrchestratorState,
                                        strategy: ResponseStrategy) -> ResponseContext:
        """Prepare comprehensive response context"""
        query_context = state["query_context"]
        search_results = state["search_results"]

        # Identify narrative elements based on strategy
        narrative_elements = await self._identify_narrative_elements(
            query_context, search_results, strategy
        )

        # Find cross-references
        cross_references = await self._find_cross_references(search_results)

        # Extract pedagogical notes
        pedagogical_notes = self._extract_pedagogical_notes(query_context, search_results)

        return ResponseContext(
            query_context=query_context,
            search_results=search_results,
            selected_strategy=strategy,
            narrative_elements=narrative_elements,
            cross_references=cross_references,
            pedagogical_notes=pedagogical_notes
        )

    async def _identify_narrative_elements(self, query_context: QueryContext,
                                           search_results: List[SearchResult],
                                           strategy: ResponseStrategy) -> Dict[str, Any]:
        """Identify elements for narrative construction"""
        elements = {
            "opening": "",
            "main_points": [],
            "transitions": [],
            "conclusion": "",
            "tone": "educational"
        }

        # Determine tone based on user level and strategy
        if query_context.user_knowledge_level == "beginner":
            elements["tone"] = "friendly_educational"
        elif query_context.user_knowledge_level == "advanced":
            elements["tone"] = "academic"

        # Set narrative structure based on strategy
        if strategy == ResponseStrategy.NARRATIVE_EXPLANATION:
            elements["opening"] = "بیایید این موضوع را به صورت داستانی بررسی کنیم..."
            elements["transitions"] = ["در ادامه", "نکته جالب اینجاست که", "اما", "همچنین"]
        elif strategy == ResponseStrategy.COMPARATIVE_ANALYSIS:
            elements["opening"] = "برای درک بهتر، بیایید مقایسه کنیم..."
            elements["transitions"] = ["از سوی دیگر", "در مقابل", "در حالی که", "برخلاف"]
        elif strategy == ResponseStrategy.PEDAGOGICAL_GUIDANCE:
            elements["opening"] = "بیایید گام به گام پیش برویم..."
            elements["transitions"] = ["گام بعدی", "حالا که این را فهمیدیم", "در نهایت"]
        elif strategy == ResponseStrategy.CASUAL_CONVERSATION:  # NEW
            elements["opening"] = "خب..."
            elements["tone"] = "casual_friendly"
            elements["transitions"] = ["راستی", "ضمناً", "به هر حال"]

        # Extract main points from search results
        for result in search_results[:3]:
            if result.relevance_score > 0.7:
                elements["main_points"].append({
                    "content": result.content[:200],
                    "importance": result.relevance_score,
                    "source": result.metadata
                })

        return elements

    async def _find_cross_references(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Find connections between different search results"""
        cross_refs = []

        # Compare results pairwise
        for i in range(len(search_results)):
            for j in range(i + 1, len(search_results)):
                result1, result2 = search_results[i], search_results[j]

                # Check for common concepts
                keywords1 = set(result1.metadata.get('keywords', []))
                keywords2 = set(result2.metadata.get('keywords', []))
                common_keywords = keywords1.intersection(keywords2)

                if len(common_keywords) >= 2:
                    cross_refs.append({
                        "source1": result1.chunk_id,
                        "source2": result2.chunk_id,
                        "connection_type": "conceptual",
                        "common_elements": list(common_keywords),
                        "strength": len(common_keywords) / max(len(keywords1), len(keywords2))
                    })

                # Check for temporal connections
                page1 = result1.metadata.get('page_number')
                page2 = result2.metadata.get('page_number')
                if page1 and page2 and abs(page1 - page2) == 1:
                    cross_refs.append({
                        "source1": result1.chunk_id,
                        "source2": result2.chunk_id,
                        "connection_type": "sequential",
                        "relationship": "consecutive_pages"
                    })

        return cross_refs

    def _extract_pedagogical_notes(self, query_context: QueryContext,
                                   search_results: List[SearchResult]) -> List[str]:
        """Extract pedagogical insights"""
        notes = []

        # Based on query complexity
        if query_context.complexity > 0.8:
            notes.append("این موضوع نیاز به تفکر عمیق دارد")
        elif query_context.complexity < 0.3:
            notes.append("این یک مفهوم پایه است")

        # Based on content type
        activity_count = sum(1 for r in search_results
                             if r.metadata.get('section_type') == 'activity')
        if activity_count > 0:
            notes.append("فعالیت‌های عملی برای درک بهتر موجود است")

        # Based on user level
        if query_context.user_knowledge_level == "beginner":
            notes.append("توضیحات ساده‌تر ارائه شده است")

        return notes

    async def _generate_initial_response(self, context: ResponseContext) -> str:
        """Generate initial response based on strategy"""

        # Handle greetings specially
        if context.query_context.query_type == QueryType.GREETING:
            greetings = [
                "سلام! 👋 من دستیار هوشمند آموزش تاریخ معاصر ایران هستم.",
                "من می‌توانم در موارد زیر به شما کمک کنم:",
                "• توضیح محتوای درس‌ها و صفحات کتاب",
                "• پاسخ به سوالات و فعالیت‌ها",
                "• ارائه خلاصه و نکات کلیدی",
                "• مقایسه رویدادها و شخصیت‌های تاریخی",
                "• توضیح روابط علت و معلولی وقایع",
                "• گفتگوی دوستانه (حتی خارج از موضوعات درسی!)",
                "",
                "چه سوالی درباره تاریخ معاصر ایران دارید؟ یا شاید بخواهید کمی گپ بزنیم؟ 📚😊"
            ]
            return "\n".join(greetings)

        # Strategy-specific prompts
        strategy_prompts = {
            ResponseStrategy.DIRECT_ANSWER: self._create_direct_answer_prompt(),
            ResponseStrategy.NARRATIVE_EXPLANATION: self._create_narrative_prompt(),
            ResponseStrategy.COMPARATIVE_ANALYSIS: self._create_comparative_prompt(),
            ResponseStrategy.HISTORICAL_CONTEXT: self._create_historical_prompt(),
            ResponseStrategy.PEDAGOGICAL_GUIDANCE: self._create_pedagogical_prompt(),
        }

        prompt = strategy_prompts.get(context.selected_strategy, self._create_direct_answer_prompt())

        # Prepare content from search results
        content_parts = []
        for i, result in enumerate(context.search_results[:3], 1):
            content_parts.append(f"[منبع {i}] {result.metadata.get('source', '')}:\n{result.content}")

        combined_content = "\n\n".join(content_parts)

        try:
            chain = prompt | self.llm | StrOutputParser()
            response = await chain.ainvoke({
                "query": context.query_context.original_query,
                "content": combined_content,
                "narrative_elements": json.dumps(context.narrative_elements, ensure_ascii=False),
                "user_level": context.query_context.user_knowledge_level,
                "intent": context.query_context.intent
            })

            return response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "متأسفانه در تولید پاسخ مشکلی پیش آمد."

    def _create_direct_answer_prompt(self) -> ChatPromptTemplate:
        """Create prompt for direct answers"""
        return ChatPromptTemplate.from_messages([
            ("system", """شما یک معلم تاریخ معاصر ایران هستید که پاسخ‌های مستقیم و واضح می‌دهد.

اصول پاسخ‌دهی:
1. مستقیم به سؤال پاسخ دهید
2. از اطلاعات منابع استفاده کنید
3. زبان ساده و قابل فهم
4. بدون مقدمه‌چینی اضافی
5. اگه کاربر متن عینی یک صفحه خواست محتوای اصلی رو کپی کنین

سطح کاربر: {user_level}
هدف کاربر: {intent}"""),
            ("human", """سؤال: {query}

محتوای منابع:
{content}""")
        ])

    def _create_narrative_prompt(self) -> ChatPromptTemplate:
        """Create prompt for narrative explanations"""
        return ChatPromptTemplate.from_messages([
            ("system", """شما یک راوی ماهر تاریخ هستید که داستان‌های آموزشی جذاب می‌گوید.

اصول روایت:
1. با یک مقدمه جذاب شروع کنید
2. داستان را به صورت منسجم روایت کنید
3. از عناصر روایی استفاده کنید: {narrative_elements}
4. با یک نتیجه‌گیری قوی پایان دهید

سطح کاربر: {user_level}"""),
            ("human", """سؤال: {query}

محتوای منابع:
{content}

عناصر روایی:
{narrative_elements}""")
        ])

    def _create_comparative_prompt(self) -> ChatPromptTemplate:
        """Create prompt for comparative analysis"""
        return ChatPromptTemplate.from_messages([
            ("system", """شما متخصص تحلیل مقایسه‌ای در تاریخ هستید.

اصول مقایسه:
1. شباهت‌ها و تفاوت‌ها را مشخص کنید
2. از جداول یا لیست‌ها استفاده کنید
3. نتیجه‌گیری واضح ارائه دهید
4. بر اساس شواهد تاریخی مقایسه کنید

سطح کاربر: {user_level}"""),
            ("human", """سؤال: {query}

محتوای منابع:
{content}""")
        ])

    def _create_historical_prompt(self) -> ChatPromptTemplate:
        """Create prompt for historical context"""
        return ChatPromptTemplate.from_messages([
            ("system", """شما مورخی هستید که رویدادها را در بستر تاریخی قرار می‌دهد.

اصول:
1. پیشینه تاریخی را توضیح دهید
2. علل و عوامل را بررسی کنید
3. پیامدها را تحلیل کنید
4. ارتباط با زمان حال را نشان دهید

سطح کاربر: {user_level}"""),
            ("human", """سؤال: {query}

محتوای منابع:
{content}""")
        ])

    def _create_pedagogical_prompt(self) -> ChatPromptTemplate:
        """Create prompt for pedagogical guidance"""
        return ChatPromptTemplate.from_messages([
            ("system", """شما مربی آموزشی هستید که یادگیری گام‌به‌گام را هدایت می‌کند.

اصول:
1. مفاهیم را به بخش‌های کوچک تقسیم کنید
2. از آسان به سخت پیش بروید
3. مثال‌های عملی ارائه دهید
4. تمرین و فعالیت پیشنهاد دهید

سطح کاربر: {user_level}
هدف آموزشی: {intent}"""),
            ("human", """سؤال: {query}

محتوای منابع:
{content}""")
        ])

    async def _enhance_response(self, initial_response: str,
                                context: ResponseContext) -> str:
        """Enhance response with contextual information"""
        enhanced = initial_response

        # Add cross-references if meaningful
        if context.cross_references:
            strong_refs = [ref for ref in context.cross_references
                           if ref.get('strength', 0) > 0.5]

            if strong_refs and context.selected_strategy != ResponseStrategy.DIRECT_ANSWER:
                connections = []
                for ref in strong_refs[:2]:  # Top 2 connections
                    if ref['connection_type'] == 'conceptual':
                        elements = ', '.join(ref['common_elements'][:3])
                        connections.append(f"ارتباط مفهومی در {elements}")
                    elif ref['connection_type'] == 'sequential':
                        connections.append("ادامه در صفحه بعد")

                if connections:
                    enhanced += f"\n\n💡 **نکته:** {' | '.join(connections)}"

        # Add context window information if relevant
        if context.query_context.query_type == QueryType.PAGE_BASED:
            for result in context.search_results:
                if result.context_window:
                    if result.context_window.get('previous'):
                        enhanced = f"🔸 *از صفحه قبل:* {result.context_window['previous'][:100]}...\n\n" + enhanced
                    if result.context_window.get('next'):
                        enhanced += f"\n\n🔹 *در صفحه بعد:* {result.context_window['next'][:100]}..."

        return enhanced

    async def _assess_response_quality(self, response: str,
                                       state: OrchestratorState) -> float:
        """Assess quality of generated response"""
        quality_factors = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "clarity": 0.0,
            "pedagogical_value": 0.0,
            "engagement": 0.0
        }

        # For casual chat, use different criteria
        if state["query_context"].is_casual_chat:
            return self._assess_casual_chat_quality(response)

        # Completeness - بررسی دقیق‌تر
        response_length = len(response.strip())
        if response_length > 100:
            quality_factors["completeness"] = min(1.0, response_length / 500)

        # بررسی آیا به سوال پاسخ داده شده
        query_keywords = set(state["query_context"].normalized_query.lower().split())
        response_keywords = set(response.lower().split())
        keyword_overlap = len(query_keywords.intersection(response_keywords))
        if keyword_overlap > 0:
            quality_factors["completeness"] = min(1.0, quality_factors["completeness"] + 0.2)

        # Accuracy - بررسی استفاده از منابع
        search_results = state.get("search_results", [])
        if search_results:
            # اگر نتایج جستجو وجود دارد
            if any(r.relevance_score > 0.7 for r in search_results[:3]):
                quality_factors["accuracy"] = 0.8
            else:
                quality_factors["accuracy"] = 0.5
        elif state["query_context"].query_type == QueryType.GREETING:
            # برای سلام نیازی به منبع نیست
            quality_factors["accuracy"] = 1.0

        # Clarity - وضوح پاسخ
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 < avg_sentence_length < 30:
                quality_factors["clarity"] = 0.8
            elif 5 < avg_sentence_length < 40:
                quality_factors["clarity"] = 0.6
            else:
                quality_factors["clarity"] = 0.4

        # Pedagogical value - ارزش آموزشی
        educational_markers = [
            'به عبارت دیگر', 'مثال', 'یعنی', 'توجه کنید',
            'نکته', 'توضیح', 'درس', 'فعالیت', 'یادگیری',
            'مفهوم', 'تعریف', 'ویژگی', 'اهمیت'
        ]
        marker_count = sum(1 for marker in educational_markers if marker in response)
        quality_factors["pedagogical_value"] = min(1.0, marker_count * 0.15)

        # Engagement - جذابیت
        engagement_indicators = ['📚', '💡', '🔸', '✨', '📖', '🎓', '👈', '⭐', '🔍']
        if any(emoji in response for emoji in engagement_indicators):
            quality_factors["engagement"] = 0.7
        if "**" in response or "*" in response:  # فرمت‌دهی
            quality_factors["engagement"] = min(1.0, quality_factors["engagement"] + 0.2)

        # Calculate weighted average
        weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "clarity": 0.2,
            "pedagogical_value": 0.2,
            "engagement": 0.1
        }

        total_score = sum(quality_factors[factor] * weight
                          for factor, weight in weights.items())

        logger.info(f"Quality assessment: {quality_factors}, Total: {total_score:.2f}")

        return total_score

    def _assess_casual_chat_quality(self, response: str) -> float:
        """Assess quality for casual chat responses - NEW"""
        quality_score = 0.5  # Base score

        # Length check
        if 10 < len(response) < 500:
            quality_score += 0.2

        # Friendliness indicators
        friendly_indicators = ['😊', '😄', '👋', '❤️', '💙', '✨', 'خوشحالم', 'عزیز', 'دوست']
        if any(indicator in response for indicator in friendly_indicators):
            quality_score += 0.2

        # Natural language
        if any(word in response for word in ['خب', 'راستی', 'ضمناً', 'به هر حال']):
            quality_score += 0.1

        return min(1.0, quality_score)

    async def _refine_response(self, response: str,
                               context: ResponseContext) -> str:
        """Refine low-quality responses"""
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """شما در حال بهبود یک پاسخ آموزشی هستید.

مشکلات احتمالی:
- پاسخ ناقص یا مبهم
- عدم استفاده مناسب از منابع
- پیچیدگی بیش از حد
- فقدان ساختار مناسب

اصلاح کنید و پاسخ بهتری ارائه دهید."""),
            ("human", """پاسخ اولیه:
{response}

سؤال اصلی: {query}
هدف کاربر: {intent}""")
        ])

        try:
            chain = refinement_prompt | self.llm | StrOutputParser()
            refined = await chain.ainvoke({
                "response": response,
                "query": context.query_context.original_query,
                "intent": context.query_context.intent
            })

            return refined

        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return response  # Return original if refinement fails

    async def _add_pedagogical_elements(self, response: str,
                                        context: ResponseContext) -> str:
        """Add pedagogical enhancements to response"""
        final_response = response

        # Skip pedagogical elements for casual chat
        if context.selected_strategy == ResponseStrategy.CASUAL_CONVERSATION:
            return final_response

        # Add suggestions based on query type
        suggestions = []

        if context.query_context.query_type == QueryType.PAGE_BASED:
            # بررسی امن برای جلوگیری از خطا
            pages = context.query_context.entities.get('pages', [])
            if pages and isinstance(pages[0], dict):
                page_num = pages[0].get('number')
                if page_num:
                    suggestions.extend([
                        f"فعالیت‌های مرتبط با صفحه {page_num} را بررسی کنید",
                        f"خلاصه درس مربوط به این صفحه"
                    ])

        elif context.query_context.query_type == QueryType.CONCEPT_SEARCH:
            concepts = context.query_context.entities.get('concepts', [])
            if concepts:
                suggestions.append(f"مقایسه {concepts[0]} با مفاهیم مشابه")

        elif context.query_context.query_type == QueryType.ACTIVITY_BASED:
            suggestions.extend([
                "راهنمای گام‌به‌گام حل فعالیت",
                "نمونه پاسخ‌های دیگر دانش‌آموزان"
            ])

        # Add learning tips based on user level
        if context.query_context.user_knowledge_level == "beginner":
            final_response += "\n\n📝 **نکته یادگیری:** از ساده به پیچیده پیش بروید و با مثال‌ها کار کنید."
        elif context.query_context.user_knowledge_level == "advanced":
            final_response += "\n\n🎯 **چالش:** آیا می‌توانید ارتباط این موضوع را با رویدادهای معاصر پیدا کنید؟"

        # Add suggestions if any
        if suggestions:
            final_response += "\n\n**پیشنهادات بعدی:**"
            for suggestion in suggestions[:3]:
                final_response += f"\n• {suggestion}"

        return final_response

    def _generate_response_metadata(self, context: ResponseContext,
                                    quality_score: float) -> Dict[str, Any]:
        """Generate comprehensive response metadata"""
        metadata = {
            "strategy_used": context.selected_strategy.value,
            "quality_score": round(quality_score, 2),
            "response_time": datetime.now().isoformat(),
            "sources_used": len(context.search_results),
            "cross_references_found": len(context.cross_references),
            "pedagogical_notes": context.pedagogical_notes,
            "difficulty_assessment": self._assess_overall_difficulty(context),
            "key_concepts": self._extract_key_concepts(context),
            "suggested_next_topics": self._suggest_next_topics(context),
            "is_casual_chat": context.query_context.is_casual_chat  # NEW
        }

        # Add source details
        if context.search_results:
            metadata["primary_sources"] = [
                {
                    "type": result.metadata.get("section_type", "unknown"),
                    "location": f"صفحه {result.metadata.get('page_number', '?')}",
                    "relevance": round(result.relevance_score, 2)
                }
                for result in context.search_results[:3]
            ]

        return metadata

    def _assess_overall_difficulty(self, context: ResponseContext) -> str:
        """Assess overall difficulty of the topic"""
        if context.query_context.is_casual_chat:
            return "آسان"

        complexities = []

        # Query complexity
        complexities.append(context.query_context.complexity)

        # Content complexity
        for result in context.search_results[:3]:
            difficulty = result.metadata.get('difficulty_level', 'medium')
            difficulty_score = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}.get(difficulty, 0.5)
            complexities.append(difficulty_score)

        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.5

        if avg_complexity < 0.4:
            return "آسان"
        elif avg_complexity < 0.7:
            return "متوسط"
        else:
            return "دشوار"

    def _extract_key_concepts(self, context: ResponseContext) -> List[str]:
        """Extract key concepts from the interaction"""
        if context.query_context.is_casual_chat:
            return []

        concepts = set()

        # From query entities
        for entity_list in context.query_context.entities.values():
            if isinstance(entity_list, list):
                for entity in entity_list:
                    if isinstance(entity, str):
                        concepts.add(entity)
                    elif isinstance(entity, dict) and 'text' in entity:
                        concepts.add(entity['text'])

        # From search results
        for result in context.search_results[:3]:
            keywords = result.metadata.get('keywords', [])
            concepts.update(keywords[:3])

        return list(concepts)[:10]  # Top 10 concepts

    def _suggest_next_topics(self, context: ResponseContext) -> List[str]:
        """Suggest related topics for further exploration"""
        if context.query_context.is_casual_chat:
            return ["سوالی از تاریخ معاصر ایران", "کمک در انجام فعالیت‌ها"]

        suggestions = []

        # Based on current topic
        if context.query_context.query_type == QueryType.LESSON_BASED:
            lesson_nums = [l['number'] for l in context.query_context.entities.get('lessons', [])]
            if lesson_nums:
                current_lesson = lesson_nums[0]
                if current_lesson < 12:  # Assuming 12 lessons
                    suggestions.append(f"درس {current_lesson + 1}")

        # Based on concepts
        concepts = self._extract_key_concepts(context)
        if concepts:
            suggestions.append(f"تحلیل عمیق‌تر {concepts[0]}")

        # Based on pedagogical goals
        if context.query_context.pedagogical_goal:
            suggestions.append(f"تمرین‌های مرتبط با {context.query_context.pedagogical_goal}")

        return suggestions[:5]


# ==================== Main RAG Graph System ====================

class EducationalRAGGraph:
    """Main educational RAG system with LangGraph orchestration"""

    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.vector_store = vector_store

        # Initialize components
        self.query_analyzer = IntelligentQueryAnalyzer(config)
        self.memory_system = GraphBasedMemory()
        self.search_engine = ContextAwareSearchEngine(config, vector_store)
        self.orchestrator = MasterOrchestrator(config)

        # Build the graph
        self.graph = self._build_graph()

        # Memory for state persistence
        self.checkpointer = MemorySaver()

        # Compile the graph
        try:
            self.app = self.graph.compile(checkpointer=self.checkpointer)
            logger.info("✅ Graph compiled successfully")
        except Exception as e:
            logger.error(f"Failed to compile graph: {e}")
            self.app = None

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("search_content", self.search_content_node)
        workflow.add_node("orchestrate_response", self.orchestrate_response_node)
        workflow.add_node("update_memory", self.update_memory_node)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        # Add edges
        workflow.add_edge("analyze_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "search_content")
        workflow.add_edge("search_content", "orchestrate_response")
        workflow.add_edge("orchestrate_response", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow

    async def analyze_query_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Analyze and understand the query"""
        query = state["query"]
        conversation_history = state.get("conversation_history", [])

        # Perform deep query analysis
        query_context = await self.query_analyzer.analyze_query(query, conversation_history)

        logger.info(f"Query analysis complete: Type={query_context.query_type.value}, "
                    f"Complexity={query_context.complexity:.2f}, "
                    f"Casual={query_context.is_casual_chat}")

        return {"query_context": query_context}

    async def retrieve_context_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Retrieve user context and prepare for search"""
        user_id = state.get("metadata", {}).get("user_id", "default")

        # Get comprehensive user context
        user_context = self.memory_system.get_user_context(user_id)

        # Update user profile with current interaction
        self.memory_system.update_user_profile(user_id, {
            "concepts": state["query_context"].entities.get("concepts", []),
            "complexity": state["query_context"].complexity,
            "is_casual_chat": state["query_context"].is_casual_chat  # NEW
        })

        return {"user_profile": user_context["profile"]}

    async def search_content_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Perform intelligent content search"""
        query_context = state["query_context"]
        user_profile = state.get("user_profile", {})

        # Check if search is needed (skip for greetings and casual chat)
        if query_context.query_type in [QueryType.GREETING, QueryType.CASUAL_CHAT]:
            return {"search_results": []}

        # Perform context-aware search
        search_results = await self.search_engine.search(
            query_context,
            {"profile": user_profile}
        )

        logger.info(f"Search complete: {len(search_results)} results found")

        return {"search_results": search_results}

    async def orchestrate_response_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Orchestrate intelligent response generation"""
        # Use master orchestrator
        result = await self.orchestrator.orchestrate_response(state)

        return {
            "final_response": result["response"],
            "metadata": result["metadata"],
            "quality_metrics": {"overall_quality": result["quality_score"]}
        }

    async def update_memory_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Update memory and learning tracking"""
        user_id = state.get("metadata", {}).get("user_id", "default")

        # Track interaction
        self.memory_system.add_interaction(
            user_id,
            state["query"],
            state["final_response"],
            {
                "query_type": state["query_context"].query_type.value,
                "topics": state["metadata"].get("key_concepts", []),
                "quality_score": state["quality_metrics"]["overall_quality"],
                "is_casual_chat": state["query_context"].is_casual_chat  # NEW
            }
        )

        # Track concept relationships (skip for casual chat)
        if not state["query_context"].is_casual_chat:
            concepts = state["metadata"].get("key_concepts", [])
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    self.memory_system.track_concept_relationship(concepts[i], concepts[j])

        # Update learning path (skip for casual chat)
        if not state["query_context"].is_casual_chat and state["query_context"].query_type in [QueryType.LESSON_BASED,
                                                                                               QueryType.ACTIVITY_BASED]:
            topic = f"{state['query_context'].query_type.value}_{state['query'][:50]}"
            if topic not in self.memory_system.learning_paths[user_id]:
                self.memory_system.learning_paths[user_id].append(topic)

        return {}

    async def chat(self, query: str, user_id: str = "default",
                   conversation_id: str = None) -> Dict[str, Any]:
        """Main chat interface"""
        start_time = time.time()

        try:
            if not self.app:
                return {
                    "response": "سیستم در حال حاضر در دسترس نیست. لطفاً کمی بعد تلاش کنید.",
                    "error": "System not initialized"
                }

            # Prepare initial state
            initial_state = OrchestratorState(
                messages=[HumanMessage(content=query)],
                query=query,
                query_context=None,
                search_results=[],
                response_context=None,
                final_response="",
                metadata={"user_id": user_id},
                quality_metrics={},
                user_profile={},
                conversation_history=[]
            )

            # Get conversation history
            if conversation_id:
                # Retrieve from memory if available
                history = self.memory_system.interaction_history.get(user_id, [])
                initial_state["conversation_history"] = [
                    {"user": h["query"], "assistant": h["response"]}
                    for h in history[-5:]  # Last 5 exchanges
                ]

            # Run the graph
            config = {"configurable": {"thread_id": conversation_id or user_id}}
            result = await self.app.ainvoke(initial_state, config)

            # Calculate timing
            elapsed_time = time.time() - start_time

            # Format response
            response_data = {
                "response": result["final_response"],
                "metadata": result.get("metadata", {}),
                "query_analysis": {
                    "type": result["query_context"].query_type.value,
                    "intent": result["query_context"].intent,
                    "complexity": result["query_context"].complexity,
                    "entities": result["query_context"].entities,
                    "is_casual_chat": result["query_context"].is_casual_chat  # NEW
                },
                "suggestions": result.get("metadata", {}).get("suggested_next_topics", []),
                "metrics": {
                    "response_time": round(elapsed_time, 3),
                    "quality_score": result.get("quality_metrics", {}).get("overall_quality", 0.0),
                    "sources_used": len(result.get("search_results", []))
                },
                "conversation_id": conversation_id or user_id
            }

            logger.info(f"Response generated in {elapsed_time:.3f}s with quality score: "
                        f"{response_data['metrics']['quality_score']:.2f}")

            return response_data

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            return {
                "response": "متأسفانه در پردازش درخواست شما مشکلی پیش آمد. لطفاً دوباره تلاش کنید.",
                "error": str(e),
                "metrics": {"response_time": time.time() - start_time}
            }


# ==================== Enhanced Vector Store ====================

class EnhancedVectorStore:
    """Production-grade vector store with advanced features"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunks = []
        self.embeddings_engine = None

        # Initialize embeddings
        try:
            self.embeddings_engine = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_key=config.openai_api_key
            )
            logger.info("Embeddings engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")

        # Indexes
        self.faiss_index = None
        self.page_index = defaultdict(list)
        self.lesson_index = defaultdict(list)
        self.activity_index = defaultdict(lambda: defaultdict(list))
        self.concept_index = defaultdict(list)

        # BM25 for keyword search
        self.bm25 = None
        self.tokenized_corpus = []

        # Cache
        self.search_cache = {}
        self.cache_size = 1000

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add chunks to vector store"""
        logger.info(f"Adding {len(chunks)} chunks to vector store")

        processed_chunks = []
        for chunk in chunks:
            processed = self._process_chunk(chunk)
            processed_chunks.append(processed)
            self.chunks.append(processed)

        # Build indexes
        self._build_indexes(processed_chunks)

        logger.info("Chunks added and indexed successfully")

    def _process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process chunk for storage"""
        # Extract metadata
        metadata = chunk.get("metadata", {})

        # Create processed chunk
        processed = {
            "chunk_id": chunk.get("id", f"chunk_{len(self.chunks)}"),
            "content": chunk.get("text_corrected", chunk.get("text", "")),
            "original_content": chunk.get("text", ""),
            "metadata": metadata,
            "keywords": chunk.get("keywords", []),
            "educational_metadata": chunk.get("educational_metadata", {}),
            "semantic_metadata": chunk.get("semantic_metadata", {}),
            "embedding": None  # Will be added later
        }

        # Update indexes
        self._update_indexes(processed, len(self.chunks))

        return processed

    def _update_indexes(self, chunk: Dict[str, Any], index: int):
        """Update various indexes"""
        metadata = chunk["metadata"]

        # Page index
        page_num = metadata.get("page_number")
        if page_num:
            self.page_index[page_num].append(index)

        # Lesson index
        lesson_num = metadata.get("lesson_number")
        if lesson_num:
            self.lesson_index[lesson_num].append(index)

        # Activity index
        if metadata.get("section_type") == "activity":
            activity_match = re.search(r'فعالیت\s*(\d+)', chunk["content"])
            if activity_match:
                activity_num = int(activity_match.group(1))
                lesson = lesson_num or 0
                self.activity_index[lesson][activity_num].append(index)

        # Concept index
        for keyword in chunk["keywords"]:
            self.concept_index[keyword.lower()].append(index)

    def _build_indexes(self, chunks: List[Dict[str, Any]]):
        """Build FAISS and BM25 indexes"""
        if not self.embeddings_engine:
            logger.warning("No embeddings engine available")
            return

        try:
            # Generate embeddings in batches
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk["content"] for chunk in batch]

                if texts:
                    embeddings = self.embeddings_engine.embed_documents(texts)

                    for j, embedding in enumerate(embeddings):
                        chunks[i + j]["embedding"] = np.array(embedding, dtype=np.float32)
                        all_embeddings.append(embedding)

            # Build FAISS index
            if all_embeddings:
                embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
                dimension = embeddings_matrix.shape[1]

                # Use HNSW for better performance
                self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
                self.faiss_index.add(embeddings_matrix)

                logger.info(f"FAISS index built with {len(all_embeddings)} vectors")

            # Build BM25 index
            self.tokenized_corpus = [chunk["content"].split() for chunk in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        except Exception as e:
            logger.error(f"Error building indexes: {e}")

    async def semantic_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search with caching"""
        # Check cache
        cache_key = hashlib.md5(f"{query}_{k}".encode()).hexdigest()
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        if not self.embeddings_engine or not self.faiss_index:
            logger.warning("Semantic search not available, falling back to keyword search")
            return self._keyword_search(query, k)

        try:
            # Generate query embedding
            query_embedding = np.array(
                self.embeddings_engine.embed_query(query),
                dtype=np.float32
            ).reshape(1, -1)

            # Search in FAISS
            distances, indices = self.faiss_index.search(query_embedding, min(k * 2, len(self.chunks)))

            # Get BM25 scores
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens) if self.bm25 else []

            # Combine results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]

                    # Calculate combined score
                    semantic_score = 1.0 / (1.0 + distance)
                    keyword_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
                    combined_score = 0.7 * semantic_score + 0.3 * keyword_score

                    results.append({
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "keywords": chunk["keywords"],
                        "score": float(combined_score),
                        "semantic_score": float(semantic_score),
                        "keyword_score": float(keyword_score)
                    })

            # Sort by combined score
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:k]

            # Cache results
            self._update_cache(cache_key, results)

            return results

        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback keyword search using BM25"""
        if not self.bm25:
            return []

        try:
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)

            # Create results with scores
            results = []
            for i, score in enumerate(scores):
                if i < len(self.chunks) and score > 0:
                    chunk = self.chunks[i]
                    results.append({
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "keywords": chunk["keywords"],
                        "score": float(score),
                        "semantic_score": 0.0,
                        "keyword_score": float(score)
                    })

            # Sort and return top k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]

        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

    def get_chunks_by_page(self, page_number: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific page"""
        indices = self.page_index.get(page_number, [])
        return [self.chunks[i] for i in indices if i < len(self.chunks)]

    def get_lesson_chunks(self, lesson_number: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific lesson"""
        indices = self.lesson_index.get(lesson_number, [])
        return [self.chunks[i] for i in indices if i < len(self.chunks)]

    def get_activity_chunks(self, activity_number: int, lesson_number: int) -> List[Dict[str, Any]]:
        """Get chunks for a specific activity"""
        indices = self.activity_index[lesson_number].get(activity_number, [])
        return [self.chunks[i] for i in indices if i < len(self.chunks)]

    def get_all_activities(self, activity_number: int) -> Dict[int, List[Dict[str, Any]]]:
        """Get an activity across all lessons"""
        results = {}
        for lesson_num, activities in self.activity_index.items():
            if activity_number in activities:
                indices = activities[activity_number]
                results[lesson_num] = [self.chunks[i] for i in indices if i < len(self.chunks)]
        return results

    def _update_cache(self, key: str, value: Any):
        """Update search cache with LRU eviction"""
        self.search_cache[key] = value

        # Evict oldest entries if cache is too large
        if len(self.search_cache) > self.cache_size:
            # Simple FIFO eviction (could be improved to LRU)
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]

    def save(self, path: str):
        """Save vector store to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save FAISS index
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(save_path / "faiss.index"))

            # Save other data
            with open(save_path / "store_data.pkl", "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "page_index": dict(self.page_index),
                    "lesson_index": dict(self.lesson_index),
                    "activity_index": {k: dict(v) for k, v in self.activity_index.items()},
                    "concept_index": dict(self.concept_index),
                    "tokenized_corpus": self.tokenized_corpus
                }, f)

            logger.info(f"Vector store saved to {path}")

        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def load(self, path: str) -> bool:
        """Load vector store from disk"""
        save_path = Path(path)

        try:
            # Load FAISS index
            faiss_path = save_path / "faiss.index"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))

            # Load other data
            data_path = save_path / "store_data.pkl"
            if data_path.exists():
                with open(data_path, "rb") as f:
                    data = pickle.load(f)

                self.chunks = data["chunks"]
                self.page_index = defaultdict(list, data["page_index"])
                self.lesson_index = defaultdict(list, data["lesson_index"])
                self.concept_index = defaultdict(list, data.get("concept_index", {}))

                # Reconstruct activity index
                self.activity_index = defaultdict(lambda: defaultdict(list))
                for lesson, activities in data["activity_index"].items():
                    for activity, indices in activities.items():
                        self.activity_index[lesson][activity] = indices

                self.tokenized_corpus = data["tokenized_corpus"]

                # Rebuild BM25
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)

                logger.info(f"Vector store loaded from {path}")
                return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")

        return False


# ==================== Main System Class ====================

class EducationalRAGSystem:
    """Main educational RAG system v5.1"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = EnhancedVectorStore(config)
        self.graph = None
        self.initialized = False

        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "casual_chat_queries": 0,  # NEW
            "educational_queries": 0,  # NEW
            "avg_response_time": 0.0,
            "avg_quality_score": 0.0,
            "query_types": Counter(),
            "error_count": 0,
            "cache_hits": 0,
            "total_sources_used": 0
        }

        # Performance monitoring
        self.response_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)

    async def initialize(self) -> bool:
        """Initialize the system"""
        try:
            logger.info("Initializing Educational RAG System v5.1...")

            # Try to load existing vector store
            vector_store_path = "data/vector_store_v51"
            if self.vector_store.load(vector_store_path):
                logger.info("✅ Loaded existing vector store")
            else:
                logger.info("📦 No existing vector store found")

            # Initialize the graph
            self.graph = EducationalRAGGraph(self.config, self.vector_store)

            if self.graph.app:
                self.initialized = True
                logger.info("✅ System initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize graph")
                return False

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False

    async def load_data(self, json_file_path: str) -> bool:
        """Load data from enhanced JSON file"""
        try:
            logger.info(f"Loading data from {json_file_path}")

            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            content_blocks = data.get('content_blocks', [])
            if not content_blocks:
                logger.error("No content blocks found in file")
                return False

            # Add chunks to vector store
            self.vector_store.add_chunks(content_blocks)

            # Save vector store
            self.vector_store.save("data/vector_store_v51")

            logger.info(f"✅ Successfully loaded {len(content_blocks)} blocks")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return False

    async def chat(self, query: str, user_id: str = "default",
                   conversation_id: str = None) -> Dict[str, Any]:
        """Main chat interface"""
        start_time = time.time()

        try:
            # Update metrics
            self.metrics["total_queries"] += 1

            if not self.initialized or not self.graph:
                self.metrics["error_count"] += 1
                return {
                    "response": "سیستم در حال حاضر آماده نیست. لطفاً کمی صبر کنید و دوباره تلاش کنید.",
                    "error": "System not initialized"
                }

            # Process through graph
            result = await self.graph.chat(query, user_id, conversation_id)

            # Update metrics based on chat type
            if result.get("query_analysis", {}).get("is_casual_chat", False):
                self.metrics["casual_chat_queries"] += 1
            else:
                self.metrics["educational_queries"] += 1

            # Update metrics
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.metrics["avg_response_time"] = sum(self.response_times) / len(self.response_times)

            quality_score = result.get("metrics", {}).get("quality_score", 0.0)
            self.quality_scores.append(quality_score)
            self.metrics["avg_quality_score"] = sum(self.quality_scores) / len(self.quality_scores)

            query_type = result.get("query_analysis", {}).get("type", "unknown")
            self.metrics["query_types"][query_type] += 1

            self.metrics["total_sources_used"] += result.get("metrics", {}).get("sources_used", 0)

            return result

        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            self.metrics["error_count"] += 1

            return {
                "response": "متأسفم، مشکلی پیش آمد. لطفاً دوباره تلاش کنید.",
                "error": str(e),
                "metrics": {"response_time": time.time() - start_time}
            }

    def add_feedback(self, query: str, response: str, rating: int,
                     user_id: str = "default", metadata: Dict[str, Any] = None):
        """Process user feedback"""
        try:
            # Track feedback in memory system
            if self.graph:
                self.graph.memory_system.update_user_profile(user_id, {
                    "last_feedback": rating,
                    "feedback_timestamp": datetime.now()
                })

                # Adjust success rate
                profile = self.graph.memory_system.user_profiles.get(user_id, {})
                current_success = profile.get("success_rate", 1.0)
                # Simple exponential moving average
                new_success = 0.9 * current_success + 0.1 * (1.0 if rating >= 4 else 0.0)
                profile["success_rate"] = new_success

            logger.info(f"Feedback recorded: Rating={rating} for user={user_id}")

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")

    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        analytics = {
            "system_metrics": {
                "total_queries": self.metrics["total_queries"],
                "educational_queries": self.metrics["educational_queries"],
                "casual_chat_queries": self.metrics["casual_chat_queries"],  # NEW
                "casual_chat_ratio": round(self.metrics["casual_chat_queries"] / max(self.metrics["total_queries"], 1),
                                           2),  # NEW
                "avg_response_time": round(self.metrics["avg_response_time"], 3),
                "avg_quality_score": round(self.metrics["avg_quality_score"], 2),
                "error_rate": self.metrics["error_count"] / max(self.metrics["total_queries"], 1),
                "cache_efficiency": self.metrics["cache_hits"] / max(self.metrics["total_queries"], 1)
            },
            "query_distribution": dict(self.metrics["query_types"]),
            "performance_trends": {
                "recent_response_times": list(self.response_times)[-10:],
                "recent_quality_scores": list(self.quality_scores)[-10:]
            },
            "content_metrics": {
                "total_chunks": len(self.vector_store.chunks),
                "indexed_pages": len(self.vector_store.page_index),
                "indexed_lessons": len(self.vector_store.lesson_index),
                "unique_concepts": len(self.vector_store.concept_index)
            },
            "system_health": {
                "status": "operational" if self.initialized else "not_initialized",
                "vector_store_ready": self.vector_store.faiss_index is not None,
                "graph_ready": self.graph is not None and self.graph.app is not None,
                "memory_usage": self._get_memory_usage(),
                "casual_chat_enabled": self.config.enable_casual_chat  # NEW
            }
        }

        # Add user analytics if available
        if self.graph and self.graph.memory_system:
            analytics["user_metrics"] = {
                "active_users": len(self.graph.memory_system.user_profiles),
                "total_interactions": sum(
                    len(interactions)
                    for interactions in self.graph.memory_system.interaction_history.values()
                ),
                "concept_graph_size": len(self.graph.memory_system.concept_graph)
            }

        return analytics

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        }


# ==================== FastAPI Application ====================

# Pydantic models for API
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query in Persian")
    user_id: str = Field("default", description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID")


class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    query_analysis: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    conversation_id: str


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    user_id: str = Field("default")
    metadata: Optional[Dict[str, Any]] = None


# Create FastAPI app
app = FastAPI(
    title="Educational RAG System API v5.1",
    description="Advanced educational chatbot with intelligent orchestration, narrative generation, and casual chat support",
    version="5.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_system: Optional[EducationalRAGSystem] = None
config: Optional[RAGConfig] = None


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global rag_system, config

    logger.info("🚀 Starting Educational RAG System v5.1...")

    # Load configuration
    config_path = Path("config.json")
    if not config_path.exists():
        # Create default config
        default_config = {
            "openai_api_key": os.environ.get("OPENAI_API_KEY", "sk-YOUR_API_KEY_HERE"),
            "primary_model": "gpt-4.1",
            "fast_model": "gpt-4.1-mini",
            "embedding_model": "text-embedding-3-large",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "enable_orchestrator": True,
            "enable_narrative_mode": True,
            "enable_context_analysis": True,
            "enable_quality_assurance": True,
            "enable_casual_chat": True
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        if "OPENAI_API_KEY" not in os.environ:
            logger.warning("⚠️ Please set OPENAI_API_KEY environment variable or update config.json")
            return

    try:
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = RAGConfig(**config_dict)

        # Initialize system
        rag_system = EducationalRAGSystem(config)

        if await rag_system.initialize():
            logger.info("✅ System initialized successfully")

            # Try to load data
            data_files = list(Path('.').glob('enhanced_*.json'))
            if data_files:
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                if await rag_system.load_data(str(latest_file)):
                    logger.info("✅ Data loaded successfully")
                else:
                    logger.warning("⚠️ Failed to load data file")
        else:
            logger.error("❌ Failed to initialize system")
            rag_system = None

    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        rag_system = None


# @app.get("/")
# async def root():
#     """Root endpoint with system information"""
#     return {
#         "name": "Educational RAG System v5.1",
#         "version": "5.1.0",
#         "description": "Advanced educational chatbot with intelligent orchestration and casual chat support",
#         "features": [
#             "Master Orchestrator for response coordination",
#             "Context-aware narrative generation",
#             "Dynamic response strategies",
#             "Cross-page relationship detection",
#             "Advanced pedagogical reasoning",
#             "Graph-based memory system",
#             "Casual chat mode for friendly conversations",
#             "Enhanced Persian number recognition",
#             "Improved query understanding",
#             "Production-grade error handling"
#         ],
#         "status": "operational" if rag_system and rag_system.initialized else "not_initialized",
#         "endpoints": {
#             "chat": "/chat",
#             "feedback": "/feedback",
#             "analytics": "/analytics",
#             "health": "/health"
#         }
#     }

# === CHANGE 1: The root endpoint now serves your HTML file ===
from fastapi.responses import FileResponse

@app.get("/", response_class=FileResponse)
async def read_index():
    """Serves the main index.html file"""
    return FileResponse('static/index.html')


# === CHANGE 2: Your original root endpoint is moved to /info ===
@app.get("/info")
async def info():
    """Root endpoint with system information"""
    return {
        "name": "Educational RAG System v5.1",
        "version": "5.1.0",
        "description": "Advanced educational chatbot with intelligent orchestration and casual chat support",
        "features": [
            "Master Orchestrator for response coordination",
            "Context-aware narrative generation",
            "Dynamic response strategies",
            "Cross-page relationship detection",
            "Advanced pedagogical reasoning",
            "Graph-based memory system",
            "Casual chat mode for friendly conversations",
            "Enhanced Persian number recognition",
            "Improved query understanding",
            "Production-grade error handling"
        ],
        "status": "operational" if rag_system else "not_initialized",
        "endpoints": {
            "chat": "/chat",
            "feedback": "/feedback",
            "analytics": "/analytics",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    if not rag_system:
        return {
            "status": "unhealthy",
            "reason": "System not initialized",
            "timestamp": datetime.now().isoformat()
        }

    health_status = {
        "status": "healthy" if rag_system.initialized else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "vector_store": rag_system.vector_store.faiss_index is not None,
            "graph": rag_system.graph is not None,
            "embeddings": rag_system.vector_store.embeddings_engine is not None,
            "casual_chat": rag_system.config.enable_casual_chat  # NEW
        },
        "metrics": {
            "total_queries": rag_system.metrics["total_queries"],
            "casual_chat_queries": rag_system.metrics["casual_chat_queries"],  # NEW
            "educational_queries": rag_system.metrics["educational_queries"],  # NEW
            "error_rate": rag_system.metrics["error_count"] / max(rag_system.metrics["total_queries"], 1),
            "avg_response_time": round(rag_system.metrics["avg_response_time"], 3)
        }
    }

    # Determine overall health
    if all(health_status["components"].values()):
        health_status["status"] = "healthy"
    elif any(health_status["components"].values()):
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"

    return health_status


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please try again later."
        )

    try:
        # Process chat request
        result = await rag_system.chat(
            query=request.query,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )

        # Ensure conversation_id is included
        if "conversation_id" not in result:
            result["conversation_id"] = request.conversation_id or request.user_id

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Collect and process user feedback"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        rag_system.add_feedback(
            query=request.query,
            response=request.response,
            rating=request.rating,
            user_id=request.user_id,
            metadata=request.metadata
        )

        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
async def analytics_endpoint():
    """Get comprehensive system analytics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        analytics = rag_system.get_analytics()
        return analytics

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-data")
async def load_data_endpoint(file_path: str = Query(..., description="Path to enhanced JSON file")):
    """Load data from enhanced JSON file"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        success = await rag_system.load_data(file_path)

        if success:
            return {
                "status": "success",
                "message": "Data loaded successfully",
                "chunks_loaded": len(rag_system.vector_store.chunks)
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load data")

    except Exception as e:
        logger.error(f"Load data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()

    user_id = f"ws_{int(time.time() * 1000)}"
    conversation_id = f"conv_{user_id}"

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "خوش آمدید به سیستم آموزشی هوشمند نسخه 5.1! 🎓✨",
            "user_id": user_id,
            "features": [
                "پاسخ‌های روایی و داستانی",
                "تحلیل عمیق محتوا",
                "راهنمایی گام‌به‌گام",
                "ارتباطات بین‌صفحه‌ای",
                "گفتگوی دوستانه و صمیمی",  # NEW
                "تشخیص بهتر اعداد فارسی"  # NEW
            ]
        })

        while True:
            # Receive message
            data = await websocket.receive_json()

            if data.get("type") == "chat":
                query = data.get("query", "")

                if not rag_system or not rag_system.initialized:
                    await websocket.send_json({
                        "type": "error",
                        "message": "سیستم در حال حاضر آماده نیست"
                    })
                    continue

                # Process query
                try:
                    result = await rag_system.chat(query, user_id, conversation_id)

                    await websocket.send_json({
                        "type": "response",
                        **result
                    })

                except Exception as e:
                    logger.error(f"WebSocket chat error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "خطا در پردازش پیام"
                    })

            elif data.get("type") == "feedback":
                # Handle feedback
                if rag_system:
                    rag_system.add_feedback(
                        query=data.get("query", ""),
                        response=data.get("response", ""),
                        rating=data.get("rating", 3),
                        user_id=user_id
                    )

                await websocket.send_json({
                    "type": "feedback_received",
                    "message": "بازخورد شما دریافت شد. متشکرم! 🙏"
                })

            elif data.get("type") == "ping":
                # Keep-alive
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "خطای سیستم"
            })
        except:
            pass
        await websocket.close()

# === CHANGE 3: Mount the static directory to serve CSS, JS, etc. ===
from fastapi.staticfiles import StaticFiles
# IMPORTANT: This must be placed at the end of the file, after all other routes.
app.mount("/static", StaticFiles(directory="static"), name="static")
# ==================== Main Entry Point ====================

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                 Educational RAG System v5.1                          ║
║                   Enhanced Edition                                   ║
║                                                                      ║
║  🎓 Advanced Multi-Agent Educational Assistant                       ║
║  🧠 Powered by LangGraph & GPT-4.1                                   ║
║  📚 Specialized for Iranian Contemporary History                     ║
║  💬 With Casual Chat Support & Enhanced Persian Numbers              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main entry point"""
    print_banner()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY") and not Path("config.json").exists():
        print("\n⚠️  Warning: OpenAI API key not found!")
        print("Please set OPENAI_API_KEY environment variable or create config.json")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  # or")
        print("  echo '{\"openai_api_key\": \"your-api-key-here\"}' > config.json")
        return

    # Check for data file
    data_files = list(Path('.').glob('enhanced_*.json'))
    if not data_files:
        print("\n⚠️  Warning: No enhanced data file found!")
        print("Please run the data processor (amar3.py) first to generate enhanced_*.json")
        print("The system will start but won't have any data to search.")

    # Configuration summary
    print("\n📋 Configuration:")
    print(f"  • Host: 127.0.0.1")
    print(f"  • Port: 8002")
    print(f"  • API Docs: http://127.0.0.1:8002/docs")
    print(f"  • WebSocket: ws://127.0.0.1:8002/ws")
    print(f"  • New Features: Casual Chat + Enhanced Persian Numbers")

    print("\n🚀 Starting server...")
    print("Press CTRL+C to stop\n")

    # Run the server
    uvicorn.run(
        "sample:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()