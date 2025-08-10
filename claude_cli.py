#!/usr/bin/env python3
import os
import sys
import argparse
from anthropic import Anthropic


def query_claude(query, model='claude-sonnet-4-20250514', api_key=None, return_usage=False):
    """
    Send a query to Claude and return the response text.
    
    Args:
        query (str): The query to send to Claude
        model (str): The Claude model to use
        api_key (str): API key (if None, will read from ANTHROPIC_API_KEY env var)
        return_usage (bool): If True, return dict with text and usage info
    
    Returns:
        str or dict: Claude's response text, or dict with 'text' and 'usage' if return_usage=True
    
    Raises:
        Exception: If API key is missing or API call fails
    """
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


def main():
    parser = argparse.ArgumentParser(description='Send queries to Anthropic Claude API')
    parser.add_argument('query', help='The query to send to Claude')
    parser.add_argument('--model', default='claude-sonnet-4-20250514', help='Claude model to use')
    
    args = parser.parse_args()
    
    try:
        response = query_claude(args.query, args.model)
        print(response)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()