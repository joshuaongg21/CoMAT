To run an evaluation, use the following command:
```
python main.py --dataset [dataset] --method [method] --model [model] --dataconfig [dataconfig]
```

For example.
```
python main.py --dataset AQUA --method symbolicot --model gpt --dataconfig normal 
```


**Datasets**
You can evaluate the following datasets:

- MMLU-Redux
- AQUA
- GSM8K
- MGSM
- Olympiad Bench
- GaoKao Mathematics

**Methods**
The evaluation can be performed using different methods:

- cot: Chain of thought reasoning, which involves multi-step reasoning processes.
- noncot: Standard question-answering without reasoning steps.
- symbolicot: Our novel approach in utilising symbolic reasoning for reasoning process.

**Models**
The following models are supported:
- gpt: gpt-4o
- gemini: gemini-1.5-pro
- llama: 70b
- qwen2: 7b and 72b

**Dataconfig (optional)**
- shuffle (shuffling the dataset options)
- swapping (randomly swap the answers and options)
- Default: normal (unchanged)
