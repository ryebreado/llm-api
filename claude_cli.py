#!/usr/bin/env python3
import os
import sys
import argparse
from anthropic import Anthropic


def main():
    parser = argparse.ArgumentParser(description='Send queries to Anthropic Claude API')
    parser.add_argument('query', help='The query to send to Claude')
    parser.add_argument('--model', default='claude-3-5-sonnet-20241022', help='Claude model to use')
    
    args = parser.parse_args()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    try:
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=args.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": args.query}
            ]
        )
        
        print(response.content[0].text)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()