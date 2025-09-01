#!/usr/bin/env python3
import argparse
import json
import os
import re
import random
from collections import defaultdict
from pathlib import Path


def clean_word(word):
    """Remove punctuation except hyphens/dashes and convert to lowercase"""
    # First preserve hyphens, en-dashes, and em-dashes
    # Then remove all other punctuation
    # Finally convert to lowercase
    cleaned = re.sub(r'[^\w\-–—]', '', word)  # Keep letters, numbers, and dash variants
    cleaned = re.sub(r'[0-9]', '', cleaned)   # Remove numbers
    return cleaned.lower()


def count_letter_occurrences(word, letter):
    """Count how many times a letter appears in a word"""
    return word.lower().count(letter.lower())


def analyze_text_file(file_path, max_n=3, max_words_per_group=3):
    """
    Analyze a text file and create a dictionary mapping letters to occurrence counts to word sets.
    
    Args:
        file_path (str): Path to the text file
        max_n (int): Maximum number of occurrences to track (0 to max_n)
        max_words_per_group (int): Maximum number of words to include per letter/count combination
    
    Returns:
        dict: Nested dictionary {letter: {count: [word_list]}}
    """
    # Initialize the nested dictionary structure
    letter_analysis = {}
    
    # Initialize for all letters a-z
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        letter_analysis[letter] = {}
        for n in range(max_n + 1):
            letter_analysis[letter][n] = set()
    
    # Read and process the text file
    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Extract words and clean them
    words = text.split()
    unique_words = set()
    
    for word in words:
        cleaned_word = clean_word(word)
        if cleaned_word and len(cleaned_word) > 0:  # Skip empty strings
            unique_words.add(cleaned_word)
    
    print(f"Found {len(unique_words)} unique words")
    
    # Analyze each word for each letter
    for word in unique_words:
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            count = count_letter_occurrences(word, letter)
            
            # Only store up to max_n occurrences
            if count <= max_n:
                letter_analysis[letter][count].add(word)
    
    # Convert sets to lists, apply random sampling, and remove empty entries
    result = {}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        letter_data = {}
        for n in range(max_n + 1):
            if letter_analysis[letter][n]:  # Only include non-empty sets
                word_list = list(letter_analysis[letter][n])
                # Apply random sampling if we have more words than the limit
                if len(word_list) > max_words_per_group:
                    word_list = random.sample(word_list, max_words_per_group)
                letter_data[n] = sorted(word_list)  # Use integer key directly
        
        # Only include letters that have at least one non-empty count
        if letter_data:
            result[letter] = letter_data
    
    return result


def save_analysis_to_json(analysis, output_path):
    """Save the analysis dictionary to a JSON file"""
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, sort_keys=True)
        print(f"Analysis saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def print_sample_results(analysis, num_samples=3):
    """Print a sample of the results for verification"""
    print(f"\nSample results (showing first {num_samples} letters with data):")
    print("-" * 60)
    
    count = 0
    for letter in sorted(analysis.keys()):
        if count >= num_samples:
            break
            
        print(f"Letter '{letter}':")
        for n in sorted(analysis[letter].keys()):
            word_count = len(analysis[letter][n])
            sample_words = analysis[letter][n][:5]  # Show first 5 words
            sample_str = ', '.join(sample_words)
            if word_count > 5:
                sample_str += f", ... ({word_count} total)"
            print(f"  {n} occurrences: {sample_str}")
        print()
        count += 1


def main():
    parser = argparse.ArgumentParser(
        description='Analyze letter occurrences in text files and generate word mappings'
    )
    parser.add_argument(
        'input_file', 
        help='Path to input text file'
    )
    parser.add_argument(
        '--max-n', 
        type=int, 
        default=3, 
        help='Maximum number of letter occurrences to track (default: 3)'
    )
    parser.add_argument(
        '--max-words', 
        type=int, 
        default=3, 
        help='Maximum number of words to include per letter/count combination (default: 3)'
    )
    parser.add_argument(
        '--output-dir', 
        default='data/generated', 
        help='Output directory for JSON file (default: data/generated)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    # Generate output filename based on input filename
    input_name = Path(args.input_file).stem
    output_filename = f"{input_name}_letter_analysis.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Analyzing letter occurrences in: {args.input_file}")
    print(f"Maximum occurrences to track: {args.max_n}")
    print(f"Maximum words per group: {args.max_words}")
    print(f"Output file: {output_path}")
    print()
    
    # Perform the analysis
    analysis = analyze_text_file(args.input_file, args.max_n, args.max_words)
    
    if analysis is None:
        return 1
    
    # Print sample results
    print_sample_results(analysis)
    
    # Save to JSON
    success = save_analysis_to_json(analysis, output_path)
    
    if success:
        print(f"\nAnalysis complete! Results saved to {output_path}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())