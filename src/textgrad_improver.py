import textgrad as tg
from src.lean_generator import refine_lean_code
from src.lean_executor import identify_inconsistencies

def apply_textgrad(initial_solution, system_prompt, question, model="gpt-4o", feedback_model="claude-3-5-sonnet-20240620", max_iterations=5):
    tg.set_backward_engine(tg.get_engine(feedback_model))
    
    solution = tg.Variable(initial_solution,
                           requires_grad=True,
                           role_description="Lean code solution")
    
    loss_system_prompt = tg.Variable(system_prompt,
                                     requires_grad=False,
                                     role_description="system prompt")
    
    loss_fn = tg.TextLoss(loss_system_prompt)
    optimizer = tg.TGD([solution])
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")
        
        loss = loss_fn(solution)
        loss.backward()
        optimizer.step()
        
        with open("temp_lean.lean", "w") as f:
            f.write(solution.value)
        
        inconsistencies = identify_inconsistencies("temp_lean.lean")
        
        if not inconsistencies:
            print("No inconsistencies found. Verification successful.")
            return solution.value
        
        print("Inconsistencies detected. Refining...")
        print("Inconsistencies:", inconsistencies)
        
        refined_code = refine_lean_code(solution.value, inconsistencies, question)
        print("Refined code:", refined_code)
        
        solution.value = refined_code
    
    print("Max iterations reached. Returning the last version of the code.")
    return solution.value