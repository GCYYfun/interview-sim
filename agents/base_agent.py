from typing import List, Dict, Any, Optional
import os
from menglong.models import Model
from menglong.schemas.chat import User, Assistant, System

class BaseAgent:
    def __init__(self, name: str, role: str, model: str = "anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        self.name = name
        self.role = role
        self.model = model
        self.client = Model(default_model_id=model)
    
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str = "", thinking: bool = False) -> str:
        """
        Generates a response using MengLong.
        messages: List of dicts with 'role' (user/assistant) and 'content'.
        """
        try:
            full_response = ""
            full_thinking = ""
            
            # Construct messages for menglong using schemas
            role_map = {"user": User, "assistant": Assistant, "system": System}
            ml_messages = [System(system_prompt)] if system_prompt else []
            
            for msg in messages:
                func = role_map.get(msg["role"], User)
                ml_messages.append(func(msg["content"]))

            # Determine if we should use thinking/reasoning
            # Note: menglong handles provider-specific params via kwargs
            kwargs = {}
            if thinking:
                # Assuming the provider supports reasoning/thinking
                kwargs["include_reasonsing"] = True # Hypothetical param based on previous context or common patterns

            thinking_started = False
            response_started = False

            for event in self.client.stream_chat(
                messages=ml_messages,
                model=self.model,
                **kwargs
            ):
                if event.output.delta.reasoning:
                    if not thinking_started:
                        print("Thinking: ", end="", flush=True)
                        thinking_started = True
                    print(event.output.delta.reasoning, end="", flush=True)
                    full_thinking += event.output.delta.reasoning
                
                if event.output.delta.text:
                    if not response_started:
                        if thinking_started:
                            print("\n", end="")
                        print("Response: ", end="", flush=True)
                        response_started = True
                    print(event.output.delta.text, end="", flush=True)
                    full_response += event.output.delta.text

            if thinking:
                return (full_thinking, full_response)
            else:
                return full_response

        except Exception as e:
            print(f"Error generating response for {self.name}: {e}")
            return f"[Error: {str(e)}]"

    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement run")
