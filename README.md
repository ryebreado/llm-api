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

### FRENK dataset
English subset of the [FRENK hate speech dataset](https://huggingface.co/datasets/classla/FRENK-hate-en), tagged for offensive/nonoffensive and two categories of hate: LGBT and migrants. I would like to use the other languages later.

## Models supported
System supports Anthropic and OpenAI APIs, automatically chooses based on the model name and the dictionary provided in `llm_client.py` in the variable `MODEL_PROVIDERS`. These commands work:
```
python evaluate_llm.py --rows=1-3 --model='gpt-4o'
```
```
python evaluate_llm.py --rows=1-3 --model='claude-sonnet-4-20250514'
```