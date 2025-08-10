#!/usr/bin/env python3
import argparse
import sys
from claude_cli import query_claude
from explore_dataset import load_frenk_dataset, get_row_by_index


def parse_row_range(range_str):
    """
    Parse a row range string like '1-5' or '10-20'.
    
    Args:
        range_str (str): Range in format 'start-end'
    
    Returns:
        tuple: (start, end) both inclusive
    
    Raises:
        ValueError: If range format is invalid
    """
    try:
        start, end = map(int, range_str.split('-'))
        if start > end:
            raise ValueError(f"Start index {start} must be <= end index {end}")
        return start, end
    except ValueError as e:
        raise ValueError(f"Invalid range format '{range_str}'. Use format 'start-end' (e.g., '1-5')")


def calculate_metrics(results):
    """
    Calculate binary classification metrics and token usage.
    
    Args:
        results (list): List of evaluation results
    
    Returns:
        dict: Dictionary with metrics
    """
    # Filter out error results
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {"error": "No valid results to calculate metrics"}
    
    # Calculate confusion matrix components
    tp = sum(1 for r in valid_results if r['actual_label'] == 'Offensive' and r['llm_label'] == 'Offensive')
    tn = sum(1 for r in valid_results if r['actual_label'] == 'Acceptable' and r['llm_label'] == 'Acceptable')
    fp = sum(1 for r in valid_results if r['actual_label'] == 'Acceptable' and r['llm_label'] == 'Offensive')
    fn = sum(1 for r in valid_results if r['actual_label'] == 'Offensive' and r['llm_label'] == 'Acceptable')
    
    total = len(valid_results)
    correct = sum(1 for r in valid_results if r['is_correct'])
    
    # Calculate token usage
    total_input_tokens = sum(r.get('token_usage', {}).get('input_tokens', 0) for r in valid_results)
    total_output_tokens = sum(r.get('token_usage', {}).get('output_tokens', 0) for r in valid_results)
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "specificity": specificity,
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        },
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn
        }
    }


def evaluate_row_range(df, start_row, end_row, model='claude-sonnet-4-20250514'):
    """
    Evaluate a range of rows from the dataset.
    
    Args:
        df: The loaded dataset
        start_row (int): Start row index (inclusive)
        end_row (int): End row index (inclusive)
        model (str): Claude model to use
    
    Returns:
        list: List of evaluation results
    """
    results = []
    
    print(f"{'Row':<4} {'Actual':<10} {'Predicted':<10} {'Correct':<7} {'Text':<60}")
    print("-" * 95)
    
    for row_idx in range(start_row, end_row + 1):
        result = evaluate_single_row(df, row_idx, model)
        results.append(result)
        
        if "error" in result:
            print(f"{row_idx:<4} {'ERROR':<10} {'ERROR':<10} {'✗':<7} {result['error'][:60]:<60}")
        else:
            text_preview = result['text'][:60].replace('\n', ' ').replace('\t', ' ')
            correct_symbol = '✓' if result['is_correct'] else '✗'
            print(f"{row_idx:<4} {result['actual_label']:<10} {result['llm_label']:<10} {correct_symbol:<7} {text_preview:<60}")
    
    return results


def evaluate_single_row(df, row_index, model='claude-sonnet-4-20250514'):
    """
    Evaluate a single row from the dataset using the LLM.
    
    Args:
        df: The loaded dataset
        row_index (int): Index of the row to evaluate
        model (str): Claude model to use
    
    Returns:
        dict: Dictionary with evaluation results
    """
    try:
        row = get_row_by_index(df, row_index)
    except IndexError as e:
        return {"error": str(e)}
    
    # Create a prompt for binary classification
    prompt = f"""Please classify the following Facebook comment on topic of {row['topic']} as either "Acceptable" or "Offensive" for hate speech detection.

Text: "{row['text']}"

Respond with only one word: either "Acceptable" or "Offensive"."""

    try:
        response_data = query_claude(prompt, model, return_usage=True)
        llm_response = response_data['text'].strip()
        token_usage = response_data['usage']
        
        # Extract the actual label from dataset
        actual_label = row['binary_label']
        
        # Normalize the LLM response
        llm_label = "Acceptable" if "acceptable" in llm_response.lower() else "Offensive"
        
        # Check if prediction matches
        is_correct = llm_label == actual_label
        
        return {
            "row_index": row_index,
            "text": row['text'],
            "actual_label": actual_label,
            "llm_response": llm_response,
            "llm_label": llm_label,
            "is_correct": is_correct,
            "detailed_label": row['detailed_label'],
            "target": row['target'],
            "topic": row['topic'],
            "token_usage": token_usage
        }
        
    except Exception as e:
        return {"error": f"LLM query failed: {e}"}


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM on FRENK hate speech dataset')
    parser.add_argument('--rows', required=True, help='Row range to evaluate (e.g., "1-5" for rows 1-5 inclusive)')
    parser.add_argument('--model', default='claude-sonnet-4-20250514', help='Claude model to use')
    parser.add_argument('--dataset', default='data/FRENK-hate-en/dev.tsv', help='Path to dataset file')
    
    args = parser.parse_args()
    
    try:
        # Parse row range
        start_row, end_row = parse_row_range(args.rows)
        
        # Load dataset
        print("Loading dataset...")
        df = load_frenk_dataset(args.dataset)
        print(f"Dataset loaded: {len(df)} rows")
        
        # Validate row range
        if end_row >= len(df):
            print(f"Error: End row {end_row} exceeds dataset size {len(df)}", file=sys.stderr)
            sys.exit(1)
        
        # Evaluate row range
        print(f"\nEvaluating rows {start_row}-{end_row}...\n")
        results = evaluate_row_range(df, start_row, end_row, args.model)
        
        # Calculate and display metrics
        metrics = calculate_metrics(results)
        
        if "error" in metrics:
            print(f"\nError calculating metrics: {metrics['error']}")
            sys.exit(1)
        
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Correct Predictions: {metrics['correct_predictions']}")
        print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall (Sensitivity): {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"Specificity: {metrics['specificity']:.3f}")
        
        print("\nToken Usage:")
        tokens = metrics['token_usage']
        print(f"  Input Tokens: {tokens['total_input_tokens']:,}")
        print(f"  Output Tokens: {tokens['total_output_tokens']:,}")
        print(f"  Total Tokens: {tokens['total_tokens']:,}")
        print(f"  Average Input Tokens per Query: {tokens['total_input_tokens'] / metrics['total_samples']:.1f}")
        print(f"  Average Output Tokens per Query: {tokens['total_output_tokens'] / metrics['total_samples']:.1f}")
        
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  True Positive (Offensive → Offensive): {cm['true_positive']}")
        print(f"  True Negative (Acceptable → Acceptable): {cm['true_negative']}")
        print(f"  False Positive (Acceptable → Offensive): {cm['false_positive']}")
        print(f"  False Negative (Offensive → Acceptable): {cm['false_negative']}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()