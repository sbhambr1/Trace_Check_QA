class PromptGenerator():
    def __init__(self) -> None:
        pass
    
    def format_input_for_prompt(example, temporal_relation="before"):
        title = example['title']
        origin_text = example['origin_text']
        destination_text = example['destination_text']
        steps = example['steps']
        label = example['label']
        enumerated_steps = list(enumerate(steps, start=1))
        prompt_steps = [f"{step[0]}. {step[1]}" for step in enumerated_steps]
        prompt_steps_string = '\n'.join(prompt_steps)
        i = next((step[0] for step in enumerated_steps if step[1] == origin_text), None)
        j = next((step[0] for step in enumerated_steps if step[1] == destination_text), None)
        if i is None or j is None:
            raise ValueError("Origin or destination text not found in steps")
        
        return title, prompt_steps_string, i, j, label
    
    def get_answer_only_prompt(title, procedure, i, j, temporal_relation="before"):
        TASK = "Given a goal, a procedure to achieve that goal and a question about the steps in the procedure, you are required to answer the question in one sentence."
        GOAL = "Goal: " + title
        PROCEDURE = "Procedure: \n" + procedure
        if temporal_relation == "before":
            QUERY = f"Must Step {j} happen before Step {i}? Select between yes or no."
        else:
            QUERY = f"Must Step {i} happen after Step {j}? Select between yes or no."
        
        full_prompt = f"{TASK}\n\n{GOAL}\n\n{PROCEDURE}\n\n{QUERY}"
        return full_prompt
        
    def get_answer_explanation_prompt(title, procedure, i, j, temporal_relation="before"):
        TASK = "Given a goal, a procedure to achieve that goal and a question about the steps in the procedure, you are required to answer the question in one sentence."
        GOAL = "Goal: " + title
        PROCEDURE = "Procedure: \n" + procedure
        # QUERY_1 = f"1. Must Step {j} happen {temporal_relation} Step {i}? Select between yes or no."
        if temporal_relation == "before":
            QUERY_1 = f"1. Must Step {j} happen before Step {i}? Select between yes or no."
        else:
            QUERY_1 = f"1. Must Step {j} happen after Step {i}? Select between yes or no."
        QUERY_2 = "Explain why or why not."
        FORMAT = "Format your answer as JSON with the key value pairs \"binary_answer\": \"yes/no answer to Q1\"; \"why_answer\": \"answer to Q2\""
        
        full_prompt = f"{TASK}\n\n{GOAL}\n\n{PROCEDURE}\n\n{QUERY_1}\n{QUERY_2}\n\n{FORMAT}"
        return full_prompt

    def get_explanation_answer_prompt(title, procedure, i, j, temporal_relation="before"):
        TASK = "Given a goal, a procedure to achieve that goal and a question about the steps in the procedure, you are required to answer the question in one sentence."
        GOAL = "Goal: " + title
        PROCEDURE = "Procedure: \n" + procedure
        QUERY_1 = f"1. Explain why or why not Step {i} must happen {temporal_relation} Step {j}. Think step by step."
        # QUERY_2 = f"2. Must Step {j} happen {temporal_relation} Step {i}? Select between yes or no"
        if temporal_relation == "before":
            QUERY_2 = f"2. Must Step {j} happen before Step {i}? Select between yes or no."
        else:
            QUERY_2 = f"2. Must Step {j} happen after Step {i}? Select between yes or no."
        FORMAT = "Format your answer as JSON with the key value pairs \"why_answer\": \"answer to Q1\"; \"binary_answer\": \"yes/no answer to Q2\""
        
        full_prompt = f"{TASK}\n\n{GOAL}\n\n{PROCEDURE}\n\n{QUERY_1}\n{QUERY_2}\n\n{FORMAT}"
        return full_prompt

    def get_nl_answer_explanation_prompt(title, procedure, i, j, temporal_relation="before"):
        TASK = "Given a goal, a procedure to achieve that goal and a question about the steps in the procedure, you are required to answer the question in one sentence."
        GOAL = "Goal: " + title
        PROCEDURE = "Procedure: \n" + procedure
        # QUERY_1 = f"1. Must Step {j} happen {temporal_relation} Step {i}? Select between yes or no."
        if temporal_relation == "before":
            QUERY_1 = f"1. Must Step {j} happen before Step {i}? Select between yes or no."
        else:
            QUERY_1 = f"1. Must Step {j} happen after Step {i}? Select between yes or no."
        QUERY_2 = "Explain why or why not."
        FORMAT = "Format your answer as follows: \n\n Answer 1: yes/no \nAnswer 2: your answer in one sentence"
        
        full_prompt = f"{TASK}\n\n{GOAL}\n\n{PROCEDURE}\n\n{QUERY_1}\n{QUERY_2}\n\n{FORMAT}"
        return full_prompt