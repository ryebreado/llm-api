#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
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


def query_llm(query, model=DEFAULT_MODEL, return_usage=False, logprobs=False, temperature=1.0):
    """
    Send a query to an LLM (Claude or OpenAI) and return the response text.
    
    Args:
        query (str): The query to send to the LLM
        model (str): The model to use (auto-routes to correct API)
        return_usage (bool): If True, return dict with text and usage info
        logprobs (bool): If True, return log probabilities (OpenAI only)
        temperature (float): Sampling temperature 0.0-2.0 (OpenAI only, ignored for Anthropic)
    
    Returns:
        str or dict: LLM's response text, or dict with 'text', 'usage', and 'logprobs' if return_usage=True
    
    Raises:
        Exception: If model is unsupported, API key is missing, or API call fails
    """
    provider = _get_provider(model)
    
    # Validate logprobs usage
    if logprobs and provider != 'openai':
        raise Exception("logprobs parameter is only supported for OpenAI models (gpt-* models)")
    
    if provider == 'anthropic':
        return _query_anthropic(query, model, None, return_usage)
    elif provider == 'openai':
        return _query_openai(query, model, None, return_usage, logprobs, temperature)
    else:
        raise Exception(f"Unknown provider: {provider}")


def _query_anthropic(query, model, api_key, return_usage):
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


def _query_openai(query, model, api_key, return_usage, logprobs=False, temperature=1.0):
    """Query OpenAI API"""
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        
    if not api_key:
        raise Exception("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    
    params = {
        "model": model,
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    
    if logprobs:
        params["logprobs"] = True
        params["top_logprobs"] = 5  # Return top 5 alternative tokens
    
    response = client.chat.completions.create(**params)
    
    if return_usage or logprobs:
        result = {
            'text': response.choices[0].message.content
        }
        
        if return_usage:
            result['usage'] = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
            
        if logprobs:
            result['logprobs'] = response.choices[0].logprobs
            
        return result
    else:
        return response.choices[0].message.content


def format_logprobs_output(response_data):
    """Format logprobs output with probabilities and perplexity"""
    if 'logprobs' not in response_data:
        return
        
    logprobs_data = response_data['logprobs']
    if not logprobs_data or not logprobs_data.content:
        return
        
    print("\n" + "="*80)
    print("TOKEN ANALYSIS WITH LOG PROBABILITIES")
    print("="*80)
    
    tokens = []
    log_probs = []
    
    for i, token_data in enumerate(logprobs_data.content):
        token = token_data.token
        logprob = token_data.logprob
        linear_prob = np.exp(logprob) * 100
        
        tokens.append(token)
        log_probs.append(logprob)
        
        print(f"Token {i+1:2d}: '{token}' | logprob: {logprob:8.4f} | probability: {linear_prob:6.2f}%")
        
        # Show top alternatives if available
        if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
            print(f"         Top alternatives:")
            for j, alt_token in enumerate(token_data.top_logprobs[:4], 1):  # Show top 4 including selected
                alt_prob = np.exp(alt_token.logprob) * 100
                prefix = "→" if alt_token.token == token else " "
                print(f"           {prefix}{j}. '{alt_token.token}' ({alt_prob:5.2f}%)")
        print()
    
    # Calculate perplexity
    if log_probs:
        avg_logprob = np.mean(log_probs)
        perplexity = np.exp(-avg_logprob)
        joint_prob = np.exp(np.sum(log_probs)) * 100
        
        print("="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total tokens: {len(tokens)}")
        print(f"Average log probability: {avg_logprob:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"Joint probability (entire sequence): {joint_prob:.4f}%")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Send queries to LLM APIs (Claude or OpenAI)')
    parser.add_argument('query', help='The query to send to the LLM')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Model to use (claude-* for Anthropic, gpt-* for OpenAI)')
    parser.add_argument('--logprobs', action='store_true', default=False, help='Return log probabilities for each token (OpenAI models only)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature 0.0-2.0 (0.0=deterministic, 2.0=very random, OpenAI only)')
    
    args = parser.parse_args()
    
    try:
        if args.logprobs:
            response_data = query_llm(args.query, args.model, return_usage=True, logprobs=True, temperature=args.temperature)
            print(response_data['text'])
            format_logprobs_output(response_data)
        else:
            response = query_llm(args.query, args.model, temperature=args.temperature)
            print(response)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()