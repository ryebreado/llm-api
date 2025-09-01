#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd


def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def load_evaluation_data(json_path):
    """Load evaluation results from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def extract_confidence_data(results):
    """Extract confidence data from logprobs for visualization"""
    confidence_data = []
    
    for result in results:
        if not result.get('logprobs') or not result.get('is_correct') is not None:
            continue
            
        # Get the confidence of the selected prediction
        first_token = result['logprobs'][0] if result['logprobs'] else None
        if not first_token:
            continue
            
        confidence_data.append({
            'word': result['word'],
            'letter': result['letter'],
            'expected_count': result['expected_count'],
            'predicted_count': result.get('predicted_count'),
            'is_correct': result['is_correct'],
            'confidence': first_token['probability'],
            'word_length': len(result['word']),
            'alternatives': first_token.get('alternatives', [])
        })
    
    return confidence_data


def create_confidence_accuracy_scatter(data, model_name, output_dir):
    """Create scatter plot of confidence vs accuracy"""
    confidence_data = extract_confidence_data(data['results'])
    if not confidence_data:
        print("No confidence data available for scatter plot")
        return None
    
    df = pd.DataFrame(confidence_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different colors for expected counts
    for count in sorted(df['expected_count'].unique()):
        mask = df['expected_count'] == count
        subset = df[mask]
        
        # Separate correct and incorrect predictions
        correct = subset[subset['is_correct'] == True]
        incorrect = subset[subset['is_correct'] == False]
        
        plt.scatter(correct['confidence'], [1] * len(correct), 
                   alpha=0.6, s=60, label=f'{count} letters (correct)',
                   marker='o')
        plt.scatter(incorrect['confidence'], [0] * len(incorrect),
                   alpha=0.6, s=60, label=f'{count} letters (incorrect)', 
                   marker='x')
    
    plt.xlabel('Model Confidence (%)')
    plt.ylabel('Prediction Outcome')
    plt.title(f'Confidence vs Accuracy - {model_name}')
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'confidence_accuracy_{model_name.replace("-", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_letter_performance_grid(data, model_name, output_dir):
    """Create grid showing accuracy by letter and expected count"""
    results = data['results']
    
    # Organize data by letter and expected count
    letter_data = defaultdict(lambda: defaultdict(list))
    for result in results:
        if result.get('is_correct') is not None:
            letter_data[result['letter']][result['expected_count']].append(result['is_correct'])
    
    # Calculate accuracy for each letter/count combination
    accuracy_matrix = []
    letters = sorted(letter_data.keys())
    max_count = max(max(counts.keys()) for counts in letter_data.values()) if letter_data else 3
    counts = list(range(max_count + 1))
    
    for letter in letters:
        row = []
        for count in counts:
            if count in letter_data[letter] and letter_data[letter][count]:
                accuracy = np.mean(letter_data[letter][count])
                sample_size = len(letter_data[letter][count])
            else:
                accuracy = np.nan
                sample_size = 0
            row.append((accuracy, sample_size))
        accuracy_matrix.append(row)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(max_count + 2, len(letters) // 2 + 2))
    
    # Prepare data for heatmap (just accuracy values)
    heatmap_data = [[cell[0] for cell in row] for row in accuracy_matrix]
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=[f'{c} letters' for c in counts],
                yticklabels=letters,
                annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0.5, vmin=0, vmax=1,
                ax=ax, cbar_kws={'label': 'Accuracy'})
    
    plt.title(f'Letter Counting Accuracy by Letter and Expected Count - {model_name}')
    plt.xlabel('Expected Letter Count')
    plt.ylabel('Letter')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'letter_performance_{model_name.replace("-", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_probability_distribution_plot(data, model_name, output_dir):
    """Create violin plot of probability distributions by expected count"""
    confidence_data = extract_confidence_data(data['results'])
    if not confidence_data:
        print("No confidence data available for probability distribution plot")
        return None
    
    df = pd.DataFrame(confidence_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create violin plot
    sns.violinplot(data=df, x='expected_count', y='confidence', 
                   hue='is_correct', split=True, inner='quart')
    
    plt.xlabel('Expected Letter Count')
    plt.ylabel('Model Confidence (%)')
    plt.title(f'Confidence Distribution by Expected Count - {model_name}')
    
    # Fix legend - let seaborn handle it properly
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Incorrect', 'Correct'], title='Prediction')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'probability_distribution_{model_name.replace("-", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_alternatives_heatmap(data, model_name, output_dir):
    """Create heatmap showing alternative predictions"""
    results = data['results']
    
    # Collect alternative predictions
    alternatives_data = defaultdict(lambda: defaultdict(int))
    prediction_counts = defaultdict(int)
    
    for result in results:
        if not result.get('logprobs') or not result['logprobs']:
            continue
            
        expected = result['expected_count']
        first_token = result['logprobs'][0]
        
        # Count main prediction
        if 'token' in first_token:
            try:
                predicted = int(first_token['token'])
                alternatives_data[expected][predicted] += 1
                prediction_counts[predicted] += 1
            except ValueError:
                pass
        
        # Count alternatives
        for alt in first_token.get('alternatives', []):
            try:
                alt_pred = int(alt['token'])
                # Weight by probability
                weight = alt['probability'] / 100.0
                alternatives_data[expected][alt_pred] += weight
                prediction_counts[alt_pred] += weight
            except ValueError:
                pass
    
    if not alternatives_data:
        print("No alternative predictions data available")
        return None
    
    # Convert to matrix format
    expected_counts = sorted(alternatives_data.keys())
    predicted_counts = sorted(prediction_counts.keys())
    
    matrix = []
    for expected in expected_counts:
        row = []
        for predicted in predicted_counts:
            count = alternatives_data[expected].get(predicted, 0)
            row.append(count)
        matrix.append(row)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
                xticklabels=[f'Predicted: {p}' for p in predicted_counts],
                yticklabels=[f'Actual: {e}' for e in expected_counts],
                annot=True, fmt='.1f', 
                cmap='Blues', 
                cbar_kws={'label': 'Frequency (weighted by probability)'})
    
    plt.title(f'Prediction Alternatives Heatmap - {model_name}')
    plt.xlabel('Model Predictions')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'alternatives_heatmap_{model_name.replace("-", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_summary_dashboard(data, model_name, output_dir):
    """Create a comprehensive dashboard with multiple metrics"""
    fig = plt.figure(figsize=(16, 12))
    
    # Extract basic statistics
    stats = data['statistics']
    results = data['results']
    
    # 1. Overall metrics (top)
    ax1 = plt.subplot(3, 3, (1, 3))
    metrics = ['Overall Accuracy', 'Success Rate', 'Accuracy (Valid Only)']
    values = [stats['overall_accuracy'], stats['success_rate'], stats['accuracy']]
    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Rate')
    ax1.set_title(f'Model Performance Summary - {model_name}')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Accuracy by expected count
    ax2 = plt.subplot(3, 3, 4)
    counts = sorted(stats['accuracy_by_count'].keys())
    accuracies = [stats['accuracy_by_count'][c]['accuracy'] for c in counts]
    sample_sizes = [stats['accuracy_by_count'][c]['total'] for c in counts]
    
    bars = ax2.bar([f'{c} letters' for c in counts], accuracies, alpha=0.7)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy by Letter Count')
    ax2.set_ylim(0, 1)
    
    # Add sample size labels
    for bar, size in zip(bars, sample_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={size}', ha='center', va='bottom', fontsize=8)
    
    # 3. Confidence distribution (if available)
    confidence_data = extract_confidence_data(results)
    if confidence_data:
        ax3 = plt.subplot(3, 3, 5)
        df = pd.DataFrame(confidence_data)
        ax3.hist([df[df['is_correct'] == True]['confidence'], 
                  df[df['is_correct'] == False]['confidence']], 
                 bins=20, alpha=0.7, label=['Correct', 'Incorrect'])
        ax3.set_xlabel('Confidence (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Confidence Distribution')
        ax3.legend()
    
    # 4. Sample size by letter (if we have letter data)
    letter_counts = defaultdict(int)
    for result in results:
        if result.get('is_correct') is not None:
            letter_counts[result['letter']] += 1
    
    if letter_counts:
        ax4 = plt.subplot(3, 3, 6)
        letters = sorted(letter_counts.keys())[:10]  # Top 10 letters
        counts = [letter_counts[l] for l in letters]
        ax4.bar(letters, counts, alpha=0.7)
        ax4.set_xlabel('Letter')
        ax4.set_ylabel('Test Cases')
        ax4.set_title('Sample Size by Letter (Top 10)')
    
    # 5. Error patterns (bottom half)
    ax5 = plt.subplot(3, 1, 3)
    
    # Count prediction errors
    error_matrix = defaultdict(lambda: defaultdict(int))
    for result in results:
        if result.get('predicted_count') is not None and not result.get('is_correct'):
            expected = result['expected_count']
            predicted = result['predicted_count']
            error_matrix[expected][predicted] += 1
    
    if error_matrix:
        # Convert to text summary
        error_text = "Common Error Patterns:\n"
        for expected in sorted(error_matrix.keys()):
            for predicted in sorted(error_matrix[expected].keys()):
                count = error_matrix[expected][predicted]
                if count > 1:  # Only show patterns that occur multiple times
                    error_text += f"Expected {expected}, predicted {predicted}: {count} times\n"
        
        ax5.text(0.05, 0.95, error_text, transform=ax5.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Error Analysis')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'summary_dashboard_{model_name.replace("-", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize letter counting evaluation results'
    )
    parser.add_argument(
        'input_json',
        help='Path to evaluation results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/generated/visualizations',
        help='Output directory for visualizations (default: data/generated/visualizations)'
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        choices=['scatter', 'grid', 'violin', 'heatmap', 'dashboard', 'all'],
        default=['all'],
        help='Which plots to generate (default: all)'
    )
    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format (default: png)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' not found")
        return 1
    
    # Load data
    print(f"Loading data from: {args.input_json}")
    data = load_evaluation_data(args.input_json)
    if data is None:
        return 1
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model name from metadata
    model_name = data.get('metadata', {}).get('model', 'Unknown')
    
    # Set up plotting style
    setup_plotting_style()
    
    print(f"Generating visualizations for model: {model_name}")
    print(f"Output directory: {args.output_dir}")
    
    # Generate requested plots
    generated_files = []
    plots_to_generate = args.plots if 'all' not in args.plots else ['scatter', 'grid', 'violin', 'heatmap', 'dashboard']
    
    for plot_type in plots_to_generate:
        try:
            if plot_type == 'scatter':
                file_path = create_confidence_accuracy_scatter(data, model_name, args.output_dir)
            elif plot_type == 'grid':
                file_path = create_letter_performance_grid(data, model_name, args.output_dir)
            elif plot_type == 'violin':
                file_path = create_probability_distribution_plot(data, model_name, args.output_dir)
            elif plot_type == 'heatmap':
                file_path = create_alternatives_heatmap(data, model_name, args.output_dir)
            elif plot_type == 'dashboard':
                file_path = create_summary_dashboard(data, model_name, args.output_dir)
            else:
                continue
                
            if file_path:
                generated_files.append(file_path)
                print(f"Generated: {file_path}")
                
        except Exception as e:
            print(f"Error generating {plot_type} plot: {e}")
    
    print(f"\nGenerated {len(generated_files)} visualization files:")
    for file_path in generated_files:
        print(f"  - {file_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())