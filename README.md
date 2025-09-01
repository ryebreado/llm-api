# llm-api
I am building a script to test multiple kinds of models on multiple datasets. I would like to support multiple APIs and multiple datasets.

## Install requirements
```
python -m pip install -r requirements.txt
```
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
The `letter_analysis.py` script analyzes text files to create mappings of letters to words containing specific numbers of that letter. It generates a nested dictionary structure like `{'a': {0: ['words', 'with', 'no', 'a'], 1: ['words', 'with', 'one', 'a']}}`.

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

## Models supported
System supports Anthropic and OpenAI APIs, automatically chooses based on the model name and the dictionary provided in `llm_client.py` in the variable `MODEL_PROVIDERS`. These commands work:
```
python evaluate_llm.py --rows=1-3 --model='gpt-4o'
```
```
python evaluate_llm.py --rows=1-3 --model='claude-sonnet-4-20250514'
```