#!/usr/bin/env python3
import argparse
import json
import os
import glob
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
                cmap='Greens', vmin=0, vmax=1,
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
    
    # Flip y-axis so Actual: 0 is at the bottom
    plt.gca().invert_yaxis()
    
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


def extract_model_name_from_filename(filename):
    """Extract model name from evaluation filename pattern"""
    # Pattern: input_evaluation_modelname_timestamp.json
    parts = Path(filename).stem.split('_')
    
    # Find 'evaluation' and extract model name after it
    try:
        eval_index = parts.index('evaluation')
        if eval_index + 1 < len(parts):
            # Join parts between 'evaluation' and timestamp (last part)
            model_parts = parts[eval_index + 1:-1]
            if model_parts:
                return '_'.join(model_parts)
    except ValueError:
        pass
    
    # Fallback: use filename stem
    return Path(filename).stem


def find_latest_files_by_model(file_paths):
    """Group files by model name and return the most recent file for each model"""
    model_files = {}
    
    for file_path in file_paths:
        model_name = extract_model_name_from_filename(file_path)
        
        if model_name not in model_files:
            model_files[model_name] = []
        model_files[model_name].append(file_path)
    
    # Get most recent file for each model (by filename timestamp or modification time)
    latest_files = {}
    for model_name, files in model_files.items():
        # Sort by modification time (most recent first)
        latest_file = max(files, key=lambda f: os.path.getmtime(f))
        latest_files[model_name] = latest_file
    
    return latest_files


def create_openai_model_comparison(model_files, letter, output_dir):
    """Create multi-OpenAI model comparison for a single letter (requires logprobs)"""
    
    # Find latest files by model name
    latest_by_model = find_latest_files_by_model(model_files)
    
    print(f"Found models: {list(latest_by_model.keys())}")
    for model, file_path in latest_by_model.items():
        print(f"  {model}: {Path(file_path).name}")
    
    # Load data from all models
    model_data = {}
    for model_name, file_path in latest_by_model.items():
        data = load_evaluation_data(file_path)
        if data:
            # Only include if logprobs are enabled
            if data.get('metadata', {}).get('logprobs_enabled', False):
                # Use extracted model name, not metadata model name
                model_data[model_name] = data
            else:
                print(f"Skipping {model_name} - no logprobs data")
    
    if len(model_data) < 2:
        print("Need at least 2 OpenAI models with logprobs data for comparison")
        return None
    
    # Filter data for the specified letter
    letter_results = {}
    for model_name, data in model_data.items():
        results = []
        for result in data['results']:
            if (result['letter'] == letter and 
                result.get('is_correct') is not None and 
                result.get('logprobs')):  # Must have logprobs
                
                confidence_data = extract_confidence_data([result])
                if confidence_data:
                    results.append({
                        'word': result['word'],
                        'expected_count': result['expected_count'],
                        'predicted_count': result.get('predicted_count'),
                        'is_correct': result['is_correct'],
                        'confidence': confidence_data[0]['confidence'],
                        'word_length': len(result['word']),
                        'alternatives': confidence_data[0].get('alternatives', [])
                    })
        letter_results[model_name] = results
    
    # Create the visualization
    fig = plt.figure(figsize=(20, 12))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # OpenAI-style colors
    
    # 1. Confidence vs Expected Count - Jittered Strip Plot
    ax1 = plt.subplot(2, 3, 1)
    
    for i, (model_name, results) in enumerate(letter_results.items()):
        if not results:
            continue
            
        df = pd.DataFrame(results)
        clean_name = model_name.replace('-', ' ').title()
        
        for count in df['expected_count'].unique():
            subset = df[df['expected_count'] == count]
            # Add jitter to y-axis
            y_jitter = count + (i - len(model_data)/2) * 0.12
            
            correct = subset[subset['is_correct'] == True]
            incorrect = subset[subset['is_correct'] == False]
            
            if len(correct) > 0:
                ax1.scatter(correct['confidence'], [y_jitter] * len(correct), 
                           alpha=0.8, s=80, color=colors[i], marker='o',
                           label=f'{clean_name} ✓' if count == df['expected_count'].min() else "")
            
            if len(incorrect) > 0:
                ax1.scatter(incorrect['confidence'], [y_jitter] * len(incorrect),
                           alpha=0.8, s=100, color=colors[i], marker='x',
                           label=f'{clean_name} ✗' if count == df['expected_count'].min() else "")
    
    ax1.set_xlabel('Model Confidence (%)')
    ax1.set_ylabel('Expected Letter Count')
    ax1.set_title(f'Confidence by Expected Count - Letter "{letter.upper()}"')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Accuracy Comparison with Confidence Intervals
    ax2 = plt.subplot(2, 3, 2)
    model_names = []
    accuracies = []
    conf_intervals = []
    
    for i, (model_name, results) in enumerate(letter_results.items()):
        if not results:
            continue
        clean_name = model_name.replace('-', ' ').title()
        model_names.append(clean_name)
        accuracy = sum(1 for r in results if r['is_correct']) / len(results)
        accuracies.append(accuracy)
        
        # 95% confidence interval
        n = len(results)
        ci = 1.96 * np.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0
        conf_intervals.append(ci)
    
    bars = ax2.bar(range(len(model_names)), accuracies, 
                   color=colors[:len(model_names)], alpha=0.8, 
                   yerr=conf_intervals, capsize=5)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names)
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy Comparison - Letter "{letter.upper()}"')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{accuracy:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confidence Distribution - Box Plot
    ax3 = plt.subplot(2, 3, 3)
    confidence_data = []
    model_labels = []
    correctness_data = []
    
    for model_name, results in letter_results.items():
        clean_name = model_name.replace('-', ' ').title()
        for result in results:
            # Only include results with confidence data
            if 'confidence' in result and result['confidence'] is not None:
                confidence_data.append(result['confidence'])
                model_labels.append(clean_name)
                correctness_data.append('Correct' if result['is_correct'] else 'Incorrect')
    
    if confidence_data:
        df_box = pd.DataFrame({
            'confidence': confidence_data,
            'model': model_labels,
            'correctness': correctness_data
        })
        
        sns.boxplot(data=df_box, x='model', y='confidence', hue='correctness', ax=ax3)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Confidence (%)')
        ax3.set_title(f'Confidence Distribution - Letter "{letter.upper()}"')
        ax3.legend(title='Prediction')
    
    # 4. Alternative Predictions Analysis
    ax4 = plt.subplot(2, 3, 4)
    
    for i, (model_name, results) in enumerate(letter_results.items()):
        clean_name = model_name.replace('-', ' ').title()
        
        # Count how often top alternative matches correct answer
        alternative_accuracy = []
        for result in results:
            if not result['is_correct'] and result['alternatives']:
                # Check if correct answer is in top alternatives
                correct_answer = str(result['expected_count'])
                alt_tokens = [alt['token'] for alt in result['alternatives'][:3]]
                alternative_accuracy.append(correct_answer in alt_tokens)
        
        if alternative_accuracy:
            alt_acc = np.mean(alternative_accuracy)
            ax4.bar(i, alt_acc, color=colors[i], alpha=0.8, label=clean_name)
            ax4.text(i, alt_acc + 0.02, f'{alt_acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Alternative Contains Correct Answer')
    ax4.set_title(f'Alternative Predictions Quality - Letter "{letter.upper()}"')
    ax4.set_ylim(0, 1)
    ax4.set_xticks(range(len(letter_results)))
    ax4.set_xticklabels([name.replace('-', ' ').title() for name in letter_results.keys()])
    
    # 5. Overconfidence Analysis
    ax5 = plt.subplot(2, 3, 5)
    
    for i, (model_name, results) in enumerate(letter_results.items()):
        clean_name = model_name.replace('-', ' ').title()
        df = pd.DataFrame(results)
        
        # Skip confidence analysis if no confidence data available
        if 'confidence' not in df.columns or df['confidence'].isna().all():
            continue
        
        # Split into high confidence (>90%) and others
        high_conf = df[df['confidence'] > 90]
        low_conf = df[df['confidence'] <= 90]
        
        if len(high_conf) > 0 and len(low_conf) > 0:
            high_acc = high_conf['is_correct'].mean()
            low_acc = low_conf['is_correct'].mean()
            
            x = [i - 0.2, i + 0.2]
            y = [high_acc, low_acc]
            labels = ['High Conf (>90%)', 'Lower Conf (≤90%)']
            
            bar1 = ax5.bar(x[0], y[0], width=0.3, color=colors[i], alpha=0.9, label='High Conf (>90%)' if i == 0 else "")
            bar2 = ax5.bar(x[1], y[1], width=0.3, color=colors[i], alpha=0.5, label='Lower Conf (≤90%)' if i == 0 else "")
            bars = [bar1[0], bar2[0]]
            
            for bar, val in zip(bars, y):
                ax5.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Accuracy')
    ax5.set_title(f'High vs Low Confidence Accuracy - Letter "{letter.upper()}"')
    ax5.set_xticks(range(len(letter_results)))
    ax5.set_xticklabels([name.replace('-', ' ').title() for name in letter_results.keys()])
    ax5.set_ylim(0, 1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='gray', alpha=0.9, label='High Confidence (>90%)'),
                      Patch(facecolor='gray', alpha=0.5, label='Lower Confidence (≤90%)')]
    ax5.legend(handles=legend_elements, loc='upper right')
    
    # 6. Calibration Analysis
    ax6 = plt.subplot(2, 3, 6)
    
    for i, (model_name, results) in enumerate(letter_results.items()):
        clean_name = model_name.replace('-', ' ').title()
        df = pd.DataFrame(results)
        
        # Skip calibration analysis if no confidence data available
        if 'confidence' not in df.columns or df['confidence'].isna().all():
            continue
        
        # Calibration bins
        confidence_bins = np.linspace(0, 100, 6)  # 5 bins
        bin_centers = []
        bin_accuracy = []
        
        for j in range(len(confidence_bins) - 1):
            mask = (df['confidence'] >= confidence_bins[j]) & (df['confidence'] < confidence_bins[j+1])
            subset = df[mask]
            
            if len(subset) >= 3:  # Minimum sample size
                bin_centers.append((confidence_bins[j] + confidence_bins[j+1]) / 2)
                bin_accuracy.append(subset['is_correct'].mean() * 100)
        
        if bin_centers:
            ax6.plot(bin_centers, bin_accuracy, 
                    marker='o', color=colors[i], linewidth=3, markersize=8,
                    label=clean_name)
    
    # Perfect calibration line
    ax6.plot([0, 100], [0, 100], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    ax6.set_xlabel('Predicted Confidence (%)')
    ax6.set_ylabel('Actual Accuracy (%)')
    ax6.set_title(f'Calibration Curve - Letter "{letter.upper()}"')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 100)
    
    plt.suptitle(f'OpenAI Models Comparison: Letter "{letter.upper()}" Counting Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    output_path = os.path.join(output_dir, f'openai_models_letter_{letter}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Visualize letter counting evaluation results'
    )
    parser.add_argument(
        'input_json',
        nargs='+',
        help='Path(s) to evaluation results JSON file(s). Supports wildcards. For comparison mode, script will automatically find the latest file for each model.'
    )
    parser.add_argument(
        '--output-dir',
        default='data/generated/visualizations',
        help='Output directory for visualizations (default: data/generated/visualizations)'
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        choices=['scatter', 'grid', 'violin', 'heatmap', 'dashboard', 'compare', 'all'],
        default=['all'],
        help='Which plots to generate. Use "compare" for OpenAI model comparison (default: all)'
    )
    parser.add_argument(
        '--letter',
        default='r',
        help='Letter to analyze for model comparison (default: r)'
    )
    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format (default: png)'
    )
    
    args = parser.parse_args()
    
    # Expand wildcards in input files
    expanded_files = []
    for pattern in args.input_json:
        matches = glob.glob(pattern)
        if matches:
            expanded_files.extend(matches)
        else:
            # If no wildcard matches, treat as literal filename
            expanded_files.append(pattern)
    
    args.input_json = expanded_files
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up plotting style
    setup_plotting_style()
    
    # Check if this is a comparison request
    if 'compare' in args.plots:
        # Validate input files exist
        valid_files = [f for f in args.input_json if os.path.exists(f)]
        if len(valid_files) < 2:
            print(f"Error: Need at least 2 valid files for comparison. Found {len(valid_files)}: {valid_files}")
            return 1
        
        print(f"Generating OpenAI model comparison for letter '{args.letter}'")
        print(f"Found {len(valid_files)} valid files")
        print(f"Output directory: {args.output_dir}")
        
        file_path = create_openai_model_comparison(valid_files, args.letter, args.output_dir)
        
        if file_path:
            print(f"Generated comparison visualization: {file_path}")
        else:
            print("Failed to generate comparison visualization")
            return 1
        
        return 0
    
    # Single model visualization mode
    if len(args.input_json) > 1:
        print("Error: Multiple files provided but 'compare' not in plots. Use --plots compare for model comparison.")
        return 1
    
    input_file = args.input_json[0]
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return 1
    
    # Load data
    print(f"Loading data from: {input_file}")
    data = load_evaluation_data(input_file)
    if data is None:
        return 1
    
    # Extract model name from metadata
    model_name = data.get('metadata', {}).get('model', 'Unknown')
    
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