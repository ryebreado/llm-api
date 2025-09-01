#!/usr/bin/env python3
import argparse
import json
import os
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from llm_client import query_llm


def load_letter_analysis_data(json_path):
    """Load the letter analysis JSON data"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


def generate_test_cases(data, max_words=None):
    """
    Generate test cases from the letter analysis data.
    
    Args:
        data (dict): Letter analysis data
        max_words (int): Limit total number of test cases for debugging
    
    Returns:
        list: Test cases with format [(letter, word, expected_count), ...]
    """
    test_cases = []
    
    for letter, counts in data.items():
        for count_str, words in counts.items():
            count = int(count_str)  # Convert string key back to int
            
            for word in words:
                test_cases.append((letter, word, count))
    
    # Limit total test cases if specified (take first k for deterministic results)
    if max_words and len(test_cases) > max_words:
        test_cases = test_cases[:max_words]
    
    return test_cases


def create_prompt(letter, word):
    """Create a prompt for letter counting"""
    return f"Answer with a single number: how many letters '{letter}' are there in '{word}'"


def evaluate_single_case(letter, word, expected_count, model, temperature=0.0, use_logprobs=False):
    """
    Evaluate a single letter counting case.
    
    Args:
        letter (str): Letter to count
        word (str): Word to analyze
        expected_count (int): Ground truth count
        model (str): Model to use
        temperature (float): Sampling temperature
        use_logprobs (bool): Whether to request logprobs
    
    Returns:
        dict: Evaluation results
    """
    prompt = create_prompt(letter, word)
    
    try:
        # Query the LLM
        if use_logprobs:
            response_data = query_llm(
                prompt, 
                model=model, 
                return_usage=True, 
                logprobs=True, 
                temperature=temperature
            )
            llm_response = response_data['text'].strip()
            usage = response_data.get('usage', {})
            logprobs_data = response_data.get('logprobs', None)
        else:
            response_data = query_llm(
                prompt, 
                model=model, 
                return_usage=True,
                temperature=temperature
            )
            if isinstance(response_data, dict):
                llm_response = response_data['text'].strip()
                usage = response_data.get('usage', {})
            else:
                llm_response = response_data.strip()
                usage = {}
            logprobs_data = None
        
        # Extract predicted count - only accept single number responses
        try:
            import re
            # Check if response is exactly a single number (with optional whitespace)
            stripped_response = llm_response.strip()
            if re.match(r'^\d+$', stripped_response):
                predicted_count = int(stripped_response)
            else:
                # Response is not a single number, mark as invalid
                predicted_count = None
        except:
            predicted_count = None
        
        # Check if prediction is correct
        is_correct = predicted_count == expected_count
        
        result = {
            'letter': letter,
            'word': word,
            'expected_count': expected_count,
            'predicted_count': predicted_count,
            'is_correct': is_correct,
            'llm_response': llm_response,
            'prompt': prompt,
            'usage': usage
        }
        
        # Add logprobs data if available
        if logprobs_data:
            result['logprobs'] = extract_logprobs_info(logprobs_data)
        
        return result
        
    except Exception as e:
        return {
            'letter': letter,
            'word': word,
            'expected_count': expected_count,
            'predicted_count': None,
            'is_correct': False,
            'llm_response': None,
            'prompt': prompt,
            'error': str(e),
            'usage': {}
        }


def extract_logprobs_info(logprobs_data):
    """Extract relevant information from logprobs data"""
    if not logprobs_data or not logprobs_data.content:
        return None
    
    tokens_info = []
    for token_data in logprobs_data.content:
        token_info = {
            'token': token_data.token,
            'logprob': token_data.logprob,
            'probability': round(min(100.0, np.exp(token_data.logprob) * 100), 2)  # Convert to percentage, cap at 100%
        }
        
        # Add top alternatives if available
        if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
            alternatives = []
            for alt in token_data.top_logprobs[:5]:  # Top 5
                alternatives.append({
                    'token': alt.token,
                    'logprob': alt.logprob,
                    'probability': round(min(100.0, np.exp(alt.logprob) * 100), 2)
                })
            token_info['alternatives'] = alternatives
        
        tokens_info.append(token_info)
    
    return tokens_info


def calculate_statistics(results):
    """Calculate overall statistics from results"""
    if not results:
        return {}
    
    total_cases = len(results)
    successful_cases = [r for r in results if r.get('predicted_count') is not None]
    correct_cases = [r for r in results if r.get('is_correct') == True]
    
    stats = {
        'total_cases': total_cases,
        'successful_predictions': len(successful_cases),
        'correct_predictions': len(correct_cases),
        'success_rate': len(successful_cases) / total_cases if total_cases > 0 else 0,
        'accuracy': len(correct_cases) / len(successful_cases) if successful_cases else 0,
        'overall_accuracy': len(correct_cases) / total_cases if total_cases > 0 else 0
    }
    
    # Count by expected count
    count_breakdown = {}
    for result in results:
        expected = result['expected_count']
        if expected not in count_breakdown:
            count_breakdown[expected] = {'total': 0, 'correct': 0}
        count_breakdown[expected]['total'] += 1
        if result.get('is_correct'):
            count_breakdown[expected]['correct'] += 1
    
    # Calculate accuracy by count
    for count, data in count_breakdown.items():
        data['accuracy'] = data['correct'] / data['total'] if data['total'] > 0 else 0
    
    stats['accuracy_by_count'] = count_breakdown
    
    return stats


def save_results(results, stats, output_path, metadata):
    """Save results and statistics to JSON file"""
    output_data = {
        'metadata': metadata,
        'statistics': stats,
        'results': results
    }
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def print_summary(stats, model):
    """Print a summary of the evaluation results"""
    print(f"\n{'='*80}")
    print(f"LETTER COUNTING EVALUATION SUMMARY - {model}")
    print(f"{'='*80}")
    
    print(f"Total test cases: {stats['total_cases']}")
    print(f"Successful predictions: {stats['successful_predictions']}")
    print(f"Correct predictions: {stats['correct_predictions']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Accuracy (of successful): {stats['accuracy']:.1%}")
    print(f"Overall accuracy: {stats['overall_accuracy']:.1%}")
    
    print(f"\nAccuracy by expected count:")
    for count in sorted(stats['accuracy_by_count'].keys()):
        data = stats['accuracy_by_count'][count]
        print(f"  {count} letters: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM letter counting abilities using generated letter analysis data'
    )
    parser.add_argument(
        'input_json',
        help='Path to letter analysis JSON file'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='Model to use for evaluation (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        help='Limit to first k test cases total for debugging'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic results)'
    )
    parser.add_argument(
        '--logprobs',
        action='store_true',
        help='Request log probabilities (OpenAI models only)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/generated',
        help='Output directory for results (default: data/generated)'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Randomly shuffle test cases before evaluation'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' not found")
        return 1
    
    print(f"Loading letter analysis data from: {args.input_json}")
    data = load_letter_analysis_data(args.input_json)
    if data is None:
        return 1
    
    print(f"Generating test cases...")
    test_cases = generate_test_cases(data, args.max_words)
    
    if args.shuffle:
        random.shuffle(test_cases)
    
    print(f"Generated {len(test_cases)} test cases")
    if args.max_words:
        print(f"Limited to first {args.max_words} test cases total")
    
    # Generate output filename
    input_name = Path(args.input_json).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace('/', '_').replace('-', '_')
    output_filename = f"{input_name}_evaluation_{model_name}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Logprobs: {'enabled' if args.logprobs else 'disabled'}")
    print(f"Output: {output_path}")
    print(f"\nStarting evaluation...")
    
    # Evaluate all test cases
    results = []
    
    # Create progress bar
    with tqdm(test_cases, desc="Evaluating", unit="case") as pbar:
        for letter, word, expected_count in pbar:
            # Update progress bar description with current word
            pbar.set_description(f"Testing '{word[:15]}...' (letter '{letter}')")
            
            result = evaluate_single_case(
                letter, word, expected_count, 
                args.model, args.temperature, args.logprobs
            )
            results.append(result)
            
            # Update postfix with running accuracy
            if len(results) > 0:
                correct = sum(1 for r in results if r.get('is_correct'))
                valid = sum(1 for r in results if r.get('predicted_count') is not None)
                accuracy = correct / valid if valid > 0 else 0
                pbar.set_postfix({
                    'Accuracy': f'{accuracy:.1%}', 
                    'Valid': f'{valid}/{len(results)}'
                })
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Prepare metadata
    metadata = {
        'input_file': args.input_json,
        'model': args.model,
        'temperature': args.temperature,
        'logprobs_enabled': args.logprobs,
        'max_words_total_limit': args.max_words,
        'shuffled': args.shuffle,
        'evaluation_timestamp': datetime.now().isoformat(),
        'total_test_cases': len(test_cases)
    }
    
    # Save results
    if save_results(results, stats, output_path, metadata):
        print(f"\nResults saved to: {output_path}")
    else:
        print(f"\nError saving results to: {output_path}")
    
    # Print summary
    print_summary(stats, args.model)
    
    return 0


if __name__ == "__main__":
    exit(main())