# llm-api
I am building a script to test multiple kinds of models on multiple datasets. I would like to support multiple APIs and multiple datasets.

## Install requirements
```
python -m pip install -r requirements.txt
```

## API Keys Setup
To use the LLM APIs, you need to set up your API keys. Create a file called `set_api_keys.sh` in the root directory:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export TOGETHERAI_API_KEY="your_together_api_key_here"
```

Then make it executable and source it before running the scripts:
```bash
chmod +x set_api_keys.sh
source set_api_keys.sh
```

**Note:** This file is already added to `.gitignore` so it won't be committed to version control. Replace the placeholder values with your actual API keys and keep them secure and private.

## Quick Start - Single Queries

You can use `llm_client.py` directly for single queries with any supported model:

```bash
# Set API keys first
source set_api_key.sh

# Basic queries
python llm_client.py "What is the capital of France?"
python llm_client.py "What is the capital of France?" --model gpt-4o
python llm_client.py "What is the capital of France?" --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# With temperature control (OpenAI and Together AI only)
python llm_client.py "Write a creative story" --model gpt-4o --temperature 1.5

# With log probabilities (OpenAI and Together AI only)
python llm_client.py "Answer with a single number: 2+2" --model gpt-4o-mini --logprobs
python llm_client.py "Answer with a single number: 2+2" --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --logprobs
```

**Parameters:**
- `--model`: Specify which model to use (auto-detects API)
- `--logprobs`: Show token probabilities and alternatives
- `--temperature`: Control randomness (0.0=deterministic, 2.0=very random)

## Datasets

We tested the infrastructure with two different datasets, but it is able to be extended to more.

### FRENK hate speech dataset
Dataset tagged for offensive/nonoffensive and two categories of hate: LGBT and migrants. In 3 languages.
* [English subset](https://huggingface.co/datasets/classla/FRENK-hate-en)
* [Slovenian subset](https://huggingface.co/datasets/classla/FRENK-hate-sl)
* [Croatian subset](https://huggingface.co/datasets/classla/FRENK-hate-hr)

#### Example run

If you run it with this command
```
python evaluate_llm.py --rows=1-3
```
You get this output:

```
Evaluating rows 1-3...

Row  Actual     Predicted  Correct Text                                                        
-----------------------------------------------------------------------------------------------
1    Offensive  Offensive  ✓       How pathetic and sick                                       
2    Acceptable Acceptable ✓       What's the 'gay agenda'?                                    
3    Acceptable Acceptable ✓       GLORIA WATSON, BRAVO, WELL SAID.                            

================================================================================
EVALUATION METRICS
================================================================================
Total Samples: 3
Correct Predictions: 3
Accuracy: 1.000 (100.0%)
Precision: 1.000
Recall (Sensitivity): 1.000
F1-Score: 1.000
Specificity: 1.000

Token Usage:
  Input Tokens: 194
  Output Tokens: 15
  Total Tokens: 209
  Average Input Tokens per Query: 64.7
  Average Output Tokens per Query: 5.0

Confusion Matrix:
  True Positive (Offensive → Offensive): 1
  True Negative (Acceptable → Acceptable): 2
  False Positive (Acceptable → Offensive): 0
  False Negative (Offensive → Acceptable): 0
```


### Letter occurrence analysis
The `letter_analysis.py` script analyzes text files to create mappings of letters to words containing specific counts of that letter. It generates a nested dictionary structure like `{'a': {0: ['words', 'with', 'no', 'a'], 1: ['words', 'with', 'one', 'a']}}`.

**Usage:**
```bash
# Analyze Alice in Wonderland text
python3 letter_analysis.py data/gutenberg/alice.txt

# Custom parameters
python3 letter_analysis.py data/gutenberg/alice.txt --max-n 4 --max-words 5
```

**Parameters:**
- `--max-n`: Maximum number of letter occurrences to track (default: 3)
- `--max-words`: Maximum words to sample per letter/count combination (default: 3)
- `--output-dir`: Directory for JSON output (default: data/generated)

The script preserves hyphenated words and outputs random samples to avoid alphabetical bias. Results are saved as JSON files in `data/generated/`.

### Letter counting evaluation
The `evaluate_letter_counting.py` script uses the generated letter analysis data to test LLM letter-counting abilities. It prompts models with questions like "Answer with a single number: how many letters 'r' are there in 'strawberry'" and compares predictions to ground truth.

**Usage:**
```bash
# Set API keys first
source set_api_key.sh

# Small test (10 cases) with logprobs
python3 evaluate_letter_counting.py data/generated/alice_letter_analysis.json --max-words 10 --model gpt-4o-mini --logprobs

# Test with Claude
python3 evaluate_letter_counting.py data/generated/alice_letter_analysis.json --max-words 20 --model claude-3-haiku-20240307

# Full evaluation with progress tracking
python3 evaluate_letter_counting.py data/generated/alice_letter_analysis.json --model gpt-4o-mini

# Compare multiple models
python3 evaluate_letter_counting.py data/generated/alice_letter_analysis.json --model gpt-4o --logprobs --max-words 50
python3 evaluate_letter_counting.py data/generated/alice_letter_analysis.json --model gpt-4o-mini --logprobs --max-words 50
```

**Parameters:**
- `--max-words`: Limit total test cases for debugging (saves API credits)
- `--model`: Model to use (supports both OpenAI and Anthropic)
- `--logprobs`: Enable log probabilities (OpenAI only)
- `--temperature`: Sampling temperature (default: 0.0)
- `--shuffle`: Randomly shuffle test cases
- `--output-dir`: Directory for results (default: data/generated)

The script includes progress bars showing real-time accuracy and validation statistics. Output filenames automatically include the model name and timestamp for easy identification (e.g., `alice_letter_analysis_evaluation_gpt_4o_mini_20250907_202449.json`). The script only counts responses that are single numbers, provides accuracy statistics by letter count, and includes logprobs data for OpenAI models showing alternative predictions and confidence levels.

### Results visualization
The `visualize_results.py` script creates comprehensive visualizations from letter counting evaluation data, focusing on models with logprobs support to show confidence and uncertainty patterns.

**Usage:**
```bash
# Generate all visualizations for single model
python3 visualize_results.py data/generated/alice_letter_analysis_evaluation_gpt_4o_mini_20250901_202042.json

# Generate specific plots
python3 visualize_results.py data/generated/evaluation_file.json --plots scatter dashboard

# Compare multiple OpenAI models with automatic model detection
python3 visualize_results.py 'data/generated/*evaluation*.json' --plots compare --letter a

# Custom output directory
python3 visualize_results.py data/generated/evaluation_file.json --output-dir results/charts
```

**Available Visualizations:**
- `scatter`: Confidence vs accuracy scatter plot - reveals overconfident vs uncertain predictions
- `grid`: Letter-specific performance heatmap - shows which letters/counts are hardest
- `violin`: Probability distribution plots - confidence patterns by expected count
- `heatmap`: Alternative predictions matrix - systematic error patterns  
- `dashboard`: Comprehensive summary with multiple metrics and error analysis
- `compare`: Multi-model comparison focusing on OpenAI models with logprobs data

**Parameters:**
- `--plots`: Select specific visualizations or 'all' (default: all)
- `--letter`: Letter to analyze for model comparison (default: r)
- `--output-dir`: Directory for generated charts (default: data/generated/visualizations)
- `--format`: Output format - png, pdf, or svg (default: png)

The script automatically extracts model names from evaluation filenames and uses the most recent evaluation file for each model when comparing multiple models. It gracefully handles missing confidence data for evaluations without logprobs. The visualizations help identify model calibration issues, systematic biases, and performance patterns across different letter counting tasks. Charts are saved as high-resolution files suitable for reports and presentations.

## Models supported
System supports Anthropic, OpenAI, and Together AI APIs, automatically chooses based on the model name prefix. These commands work:
```
python evaluate_llm.py --rows=1-3 --model='gpt-4o'
```
```
python evaluate_llm.py --rows=1-3 --model='claude-sonnet-4-20250514'
```
```
python evaluate_llm.py --rows=1-3 --model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
```

**Model routing:**
- `claude-*` → Anthropic API
- `gpt-*` → OpenAI API  
- `meta-*` or models with `/` → Together AI API

**Logprobs support:**
- OpenAI: Full logprobs with alternatives (top 5 tokens)
- Together AI: Basic logprobs (selected token only)
- Anthropic: Not supported
