import requests
import json
from typing import Dict, List, Any, Optional


# This version doesn't inherit from BaseLLM since there may be import issues
class CustomLLMClient:
    """
    Custom LLM Client for connecting to internal company LLM API
    that uses Mistral model with a different API format.
    """

    def __init__(self, api_url: str, model_name: str = "mistral-7b-inst-2252b"):
        self.api_url = api_url
        self.model_name = model_name
        self.streaming = False
        self.headers = {"Content-Type": "application/json"}

    def chat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = 1000,
            temperature: float = 0.3,
            streaming: bool = False,
            **kwargs
    ) -> str:
        """
        Send a chat message to the internal LLM API and return the response.

        Args:
            messages: List of message dictionaries with role and content
            max_tokens: Maximum tokens to generate (optional)
            temperature: Sampling temperature (0-1)
            streaming: Whether to stream the response (not supported)
            **kwargs: Additional arguments to pass to the API

        Returns:
            The LLM response as a string
        """
        if streaming:
            print(
                "Warning: Streaming is not supported in this custom LLM implementation. Falling back to non-streaming.")

        try:
            # Convert the messages list to a prompt string that the model can understand
            prompt = self._convert_messages_to_prompt(messages)

            # Call the execute_task method which handles the specific API format
            response = self.execute_task(prompt, max_tokens, temperature)
            return response

        except requests.RequestException as e:
            print(f"Error calling internal LLM API: {e}")
            # Return a fallback response
            return "I apologize, but I'm having trouble connecting to the LLM API. Please try again later."

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of chat messages to a single prompt string.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            A formatted prompt string
        """
        prompt_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt_parts.append(f"<s>[INST] System: {content} [/INST]</s>")
            elif role == "user":
                prompt_parts.append(f"<s>[INST] {content} [/INST]</s>")
            elif role == "assistant":
                prompt_parts.append(f"<s>{content}</s>")

        # Join all parts with a space
        return " ".join(prompt_parts)

    def execute_task(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """Execute a task using the LLM API"""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "parameters": {
                        "extra": {
                            "temperature": temperature,
                            "max_new_tokens": max_tokens,
                            "repetition_penalty": 1
                        }
                    },
                    "inputs": [
                        {
                            "name": "input",
                            "shape": [1],
                            "datatype": "str",
                            "data": [prompt]
                        }
                    ]
                },
                timeout=60  # Set a timeout to avoid hanging
            )

            response.raise_for_status()
            response_data = response.json()
            llm_response = response_data['outputs'][0]['data'][0]

            print("\n--- LLM Response Preview ---")
            print(llm_response[:500] + "..." if len(llm_response) > 500 else llm_response)
            print("----------------------------\n")

            return llm_response

        except Exception as e:
            print(f"Error in execute_task: {str(e)}")
            raise


# Additional method to make this work directly with CrewAI
def get_llm_callback(api_url, model_name="mistral-7b-inst-2252b"):
    """
    Returns a callback function that can be used with CrewAI's custom_llm_callback parameter.
    This avoids the need to inherit from BaseLLM.

    Usage:
    agent = Agent(
        ...
        custom_llm_callback=get_llm_callback(api_url)
    )
    """
    client = CustomLLMClient(api_url, model_name)

    def callback(messages, **kwargs):
        return client.chat(messages, **kwargs)

    return callback


def get_custom_llm(api_url, model_name="mistral-7b-inst-2252b"):
    """Helper function to create a custom LLM instance"""
    return CustomLLMClient(api_url=api_url, model_name=model_name)