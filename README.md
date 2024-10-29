# CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning

This repository contains the evaluation code for the paper "[**CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning**](https://arxiv.org/pdf/2410.10336)"

### Installation
1. Clone the repository:
```bash 
git clone https://github.com/joshuaongg21/CoMAT.git
```
2. Navigate to the project directory:
```bash 
cd COMAT
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

### Evaluation
To evaluate CoMAT and other corresponding methods, run the following command:

```bash
python main.py --dataset [dataset] --method [method] --model [model] --dataconfig [dataconfig]
```

For example.
```bash
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
- comat: Our CoMAT approach in utilising symbolic reasoning for reasoning process.

**Models**
The following models are supported:
- gpt: gpt-4o
- gemini: gemini-1.5-pro
- qwen2: 7b and 72b

**Dataconfig (optional)**
- Default: normal (unchanged)
- swapping (randomly swap the answers and options) (optional)

Alternatively, we can evaluate using the bash script below:
```bash
bash evaluate.sh
```
