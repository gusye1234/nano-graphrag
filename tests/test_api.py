"""A simple test script to verify custom Azure OpenAI API connectivity and functionality.

Useful when the Azure OpenAI endpoint paths do not conform to the public OpenAI API patterns.
Requires: .env file in the root directory, one level up from this script, with the following variables:

AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT="https://<path>/gpt-4o"
OPENAI_API_VERSION=""
AZURE_OPENAI_MINI_ENDPOINT="https://<path>/gpt-4o-mini/"
OPENAI_API_VERSION=""

Point of contact: jchae@esri.com
"""

    
from dotenv import load_dotenv
import asyncio
from openai import OpenAIError

from nano_graphrag._llm import azure_gpt_4o_complete, gpt_4o_mini_complete, azure_gpt_4o_mini_complete
from nano_graphrag._llm import get_azure_openai_async_client_instance, get_azure_openai_mini_client_instance


def test_azure_gpt_4o_complete():
    async def _test():
        try:
            azure_client = get_azure_openai_async_client_instance()
            # Try to print the actual endpoint used by the client
            api_endpoint = getattr(azure_client, "_base_url", None) or getattr(azure_client, "base_url", None) or "<unknown>"
            print("Actual Azure API endpoint used:", api_endpoint)
            response = await azure_gpt_4o_complete("Say hello to GraphRAG!")
            print("Response:", response)
        except OpenAIError as e:
            print("Error during Azure GPT-4o completion:", str(e))
            print("Check that the .env file exists and includes values for AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION.")
    asyncio.run(_test())


def test_azure_gpt_4o_mini_complete():
    async def _test():
        try:
            azure_client = get_azure_openai_async_client_instance()
            api_endpoint = getattr(azure_client, "_base_url", None) or getattr(azure_client, "base_url", None) or "<unknown>"
            print("Actual Azure API endpoint used:", api_endpoint)
            response = await azure_gpt_4o_mini_complete("Say hello to GraphRAG MINI!")
            print("Response:", response)
        except OpenAIError as e:
            print("Error during Azure GPT-4o completion:", str(e))
            print("Check that the .env file exists and includes values for AZURE_OPENAI_API_KEY, AZURE_OPENAI_MINI_ENDPOINT, and OPENAI_API_VERSION.")

    asyncio.run(_test())


if __name__ == "__main__":
    load_dotenv()

    test_azure_gpt_4o_complete()
    test_azure_gpt_4o_mini_complete()

    







    

    
    

        
