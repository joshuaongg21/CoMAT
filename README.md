To run an evaluation, use the following command:
```
python main.py --dataset [dataset] --method [method] --model [model]
```

For example.
```
python main.py --dataset AQUA --method symbolicot --model gpt
```


**Datasets**
You can evaluate the following datasets:

- GPQA
- MATH
- MMLU-Pro
- MMLU
- AQUA
- TruthfulQA

**Methods**
The evaluation can be performed using different methods:

- cot: Chain of thought reasoning, which involves multi-step reasoning processes.
- noncot: Standard question-answering without reasoning steps.
- symbolicot: Symbolic chain of thought reasoning, which combines symbolic reasoning with natural language.

**Models**
The following models are supported:
- gpt: gpt-4o
- llama: [Pending]
