# CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning

This repository contains the code for the paper **CoMAT**

### Evaluation
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
- Olympiad Bench
- GaoKao Mathematics
- MGSM

**Methods**
The evaluation can be performed using different methods:

- noncot: Standard question-answering without reasoning steps.
- cot: Chain of thought reasoning, which involves multi-step reasoning processes.
- CoMAT: Our CoMAT approach in utilising symbolic reasoning for reasoning process.

**Models**
The following models are supported:
- gpt: gpt-4o
- gemini: gemini-1.5-pro
- qwen2: 7b and 72b

**Dataconfig (optional)**
- Default: normal (unchanged)
- swapping (randomly swap the answers and options) (optional)
