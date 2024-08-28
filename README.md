To run an evaluation, use the following command:
```
python main.py --dataset [dataset] --method [method] --model [model] --dataconfig [dataconfig]
```

For example.
```
python main.py --dataset AQUA --method symbolicot --model gpt --dataconfig shuffle
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
- llama: 8b and 70b
- codestral
- qwen2: 7b
- phi-3.5

**Dataconfig**
- shuffle (shuffling the dataset options)
- swapping (randomly swap the answers and options)
- normal (unchanged)
