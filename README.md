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
```

Then make it executable and source it before running the scripts:
```bash
chmod +x set_api_keys.sh
source set_api_keys.sh
```

**Note:** This file is already added to `.gitignore` so it won't be committed to version control. Replace the placeholder values with your actual API keys and keep them secure and private.
## Example run

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

## Datasets

### FRENK hate speech dataset
Dataset tagged for offensive/nonoffensive and two categories of hate: LGBT and migrants. In 3 languages.
* [English subset](https://huggingface.co/datasets/classla/FRENK-hate-en)
* [Slovenian subset](https://huggingface.co/datasets/classla/FRENK-hate-sl)
* [Croatian subset](https://huggingface.co/datasets/classla/FRENK-hate-hr)

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

# Full evaluation
python3 evaluate_letter_counting.py data/generated/alice_letter_analysis.json --model gpt-4o-mini
```

**Parameters:**
- `--max-words`: Limit total test cases for debugging (saves API credits)
- `--model`: Model to use (supports both OpenAI and Anthropic)
- `--logprobs`: Enable log probabilities (OpenAI only)
- `--temperature`: Sampling temperature (default: 0.0)
- `--shuffle`: Randomly shuffle test cases
- `--output-dir`: Directory for results (default: data/generated)

The script only counts responses that are single numbers, provides accuracy statistics by letter count, and includes logprobs data for OpenAI models showing alternative predictions and confidence levels.

### Results visualization
The `visualize_results.py` script creates comprehensive visualizations from letter counting evaluation data, focusing on models with logprobs support to show confidence and uncertainty patterns.

**Usage:**
```bash
# Generate all visualizations
python3 visualize_results.py data/generated/alice_letter_analysis_evaluation_gpt_4o_mini_20250901_202042.json

# Generate specific plots
python3 visualize_results.py data/generated/evaluation_file.json --plots scatter dashboard

# Custom output directory
python3 visualize_results.py data/generated/evaluation_file.json --output-dir results/charts
```

**Available Visualizations:**
- `scatter`: Confidence vs accuracy scatter plot - reveals overconfident vs uncertain predictions
- `grid`: Letter-specific performance heatmap - shows which letters/counts are hardest
- `violin`: Probability distribution plots - confidence patterns by expected count
- `heatmap`: Alternative predictions matrix - systematic error patterns  
- `dashboard`: Comprehensive summary with multiple metrics and error analysis

**Parameters:**
- `--plots`: Select specific visualizations or 'all' (default: all)
- `--output-dir`: Directory for generated charts (default: data/generated/visualizations)
- `--format`: Output format - png, pdf, or svg (default: png)

The visualizations help identify model calibration issues, systematic biases, and performance patterns across different letter counting tasks. Charts are saved as high-resolution files suitable for reports and presentations.

## Models supported
System supports Anthropic and OpenAI APIs, automatically chooses based on the model name and the dictionary provided in `llm_client.py` in the variable `MODEL_PROVIDERS`. These commands work:
```
python evaluate_llm.py --rows=1-3 --model='gpt-4o'
```
```
python evaluate_llm.py --rows=1-3 --model='claude-sonnet-4-20250514'
```