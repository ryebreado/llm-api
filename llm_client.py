#!/usr/bin/env python3
import os
import sys
import argparse
from anthropic import Anthropic
from openai import OpenAI
from typing import Any

def _get_provider(model):
    """Determine provider based on model name prefix"""
    if model.startswith('claude-'):
        return 'anthropic'
    elif model.startswith('gpt-'):
        return 'openai'
    else:
        raise Exception(f"Unsupported model: {model}. Model must start with 'claude-' or 'gpt-'")

DEFAULT_MODEL = 'claude-sonnet-4-20250514'


def query_llm(query, model=DEFAULT_MODEL, return_usage=False):
    """
    Send a query to an LLM (Claude or OpenAI) and return the response text.
    
    Args:
        query (str): The query to send to the LLM
        model (str): The model to use (auto-routes to correct API)
        return_usage (bool): If True, return dict with text and usage info
    
    Returns:
        str or dict: LLM's response text, or dict with 'text' and 'usage' if return_usage=True
    
    Raises:
        Exception: If model is unsupported, API key is missing, or API call fails
    """
    provider = _get_provider(model)
    
    if provider == 'anthropic':
        return _query_anthropic(query, model, None, return_usage)
    elif provider == 'openai':
        return _query_openai(query, model, None, return_usage)
    else:
        raise Exception(f"Unknown provider: {provider}")


def _query_anthropic(query, model, api_key, return_usage) -> str | dict[str, Any]:
    """Query Anthropic Claude API"""
    if api_key is None:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY environment variable not set")
    
    client = Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    
    if return_usage:
        return {
            'text': response.content[0].text,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
        }
    else:
        return response.content[0].text


def _query_openai(query, model, api_key, return_usage):
    """Query OpenAI API"""
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        
    if not api_key:
        raise Exception("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    
    if return_usage:
        return {
            'text': response.choices[0].message.content,
            'usage': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
        }
    else:
        return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='Send queries to LLM APIs (Claude or OpenAI)')
    parser.add_argument('query', help='The query to send to the LLM')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Model to use (claude-* for Anthropic, gpt-* for OpenAI)')
    
    args = parser.parse_args()
    
    try:
        response = query_llm(args.query, args.model)
        print(response)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()