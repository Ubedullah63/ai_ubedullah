"""
test_langchain.py
Standalone test script to verify LangChain integration with Gemini 3 Flash.
Run via: python test_langchain.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    print("\n❌ Error: GEMINI_API_KEY is not set.")
    print("Please set it in your .env file or run:")
    print("  export GEMINI_API_KEY='your_api_key'\n")
    sys.exit(1)

print("\n🦜 Testing LangChain + Google Gemini 3 Flash Integration...")

try:
    from utils.langchain_service import stream_langchain_chat

    sample_telemetry = {
        "forecastHorizon": "7d",
        "totalUnitsPredicted": 88,
        "projectedRevenue": "$3.82M",
        "avgSellingPrice": "$43,409",
        "leadToDealRate": "23.4%",
        "operatingLevers": {
            "marketingDeltaPct": 15,
            "dealerDiscountPct": 5.0,
            "customerAPR": 5.9,
            "inventoryAdequacy": 95
        }
    }

    test_messages = [
        {"role": "user", "content": "Briefly evaluate how our 5.9% APR affects test-drive conversions this week."}
    ]

    print("\n[User Prompt]:", test_messages[0]["content"])
    print("[AI Streamed Response]:\n" + "-" * 50)

    for chunk in stream_langchain_chat(test_messages, sample_telemetry, api_key=api_key):
        chunk_str = str(chunk) if not isinstance(chunk, str) else chunk
        sys.stdout.write(chunk_str)
        sys.stdout.flush()

    print("\n" + "-" * 50)
    print("✅ LangChain integration passed successfully!\n")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Run: pip install langchain langchain-core langchain-google-genai\n")
except Exception as e:
    print(f"\n❌ Execution error: {e}\n")