import os
from dotenv import load_dotenv
import openai
import anthropic

# Load environment variables
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up your Anthropic API key
anthropic_client = anthropic.Client(api_key=os.getenv('CLAUDE_API_KEY'))

# Output file path
output_file_path = 'mmlupro.json'

# Prompt file paths
formulation_prompt_path = 'prompts/MMLU-Pro-Mathematics/formulation.txt'
z3_solver_prompt_path = 'prompts/MMLU-Mathematics/Z3-solver.txt'