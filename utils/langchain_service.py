"""
utils/langchain_service.py
LangChain integration module using ChatGoogleGenerativeAI and prompt chaining.
"""

import json
import os
from typing import Generator, List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


def _extract_text_from_chunk(content: Any) -> str:
    """Safely extracts plain string from chunk content whether it's str or list."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content) if content is not None else ""


def build_system_instruction(telemetry_summary: Dict[str, Any]) -> str:
    """Creates a grounded system prompt containing live dealership telemetry."""
    return f"""You are the Chief Dealership Strategist and Retail Automotive Consultant for an automotive franchise group.
You advise the General Manager (GM), General Sales Manager (GSM), and F&I Director.

CURRENT DEALERSHIP LIVE TELEMETRY & FORECAST STATE:
{json.dumps(telemetry_summary, indent=2)}

Guidelines:
1. Always reference the dealership's actual telemetry numbers (projected units, APR %, discount %, ASP, lead count) in your reasoning.
2. Be direct, authoritative, and actionable (focus on gross profit, PVR, floorplan cost, closing ratio).
3. If the user asks about changing rates or discounts, evaluate how it impacts dealer gross and inventory turn.
4. Keep formatting clean with bold bullet points where appropriate.
"""


def convert_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    """Converts Streamlit chat history format to LangChain message objects."""
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


def stream_langchain_chat(
    messages: List[Dict[str, str]],
    telemetry_summary: Dict[str, Any],
    api_key: str = "",
    model_name: str = "gemini-3-flash-preview"
) -> Generator[str, None, None]:
    """
    Streams LangChain conversational response chunk-by-chunk using ChatGoogleGenerativeAI.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError("Missing Google Gemini API key.")

    # Initialize LangChain Google GenAI Chat Model
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
        max_retries=2,
    )

    system_text = build_system_instruction(telemetry_summary)
    
    # Setup LangChain Prompt Template with System Message and conversation history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_instruction}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ])

    # Convert conversation messages excluding the current prompt
    history_objects = convert_chat_history(messages)

    # Chain execution using LangChain Expression Language (LCEL)
    chain = prompt | llm

    # Stream chunks back to Streamlit or CLI safely as string
    for chunk in chain.stream({
        "system_instruction": system_text,
        "chat_history": history_objects,
    }):
        text = _extract_text_from_chunk(chunk.content)
        if text:
            yield text


def invoke_langchain_chat(
    messages: List[Dict[str, str]],
    telemetry_summary: Dict[str, Any],
    api_key: str = "",
    model_name: str = "gemini-3-flash-preview"
) -> str:
    """
    Synchronous single-call invocation via LangChain.
    """
    chunks = list(stream_langchain_chat(messages, telemetry_summary, api_key, model_name))
    return "".join(chunks)