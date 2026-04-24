import re
import json
import os
import re

from .math_competitions.utils import mathcontest_process_results_with_feedback, aime_process_results 
from .olympiad.utils import proof_rearrangement_process_results
from .AMPS_Hard.utils import amps_hard_process_results 

def calculate_livebench_score(question_d, llm_answer, debug=False):
    question = question_d

    coding_test_case_tasks = ["coding_completion", "LCB_generation", "code_generation", "code_completion", "agentic_coding"]
    if "ground_truth" not in question and question["task"] not in coding_test_case_tasks and question["category"] != "instruction_following":
        # aside from coding and instruction following tasks, all questions should contain the ground truth answer
        raise ValueError("Questions must have ground_truth to run gen_ground_truth_judgment.")

    task = question["task"]
    task_or_subtask = question["subtask"] if "subtask" in question.keys() else question["task"]
    question_text = question["turns"][0]
    ground_truth = question.get("ground_truth", None)
    llm_answer = re.sub(r"<think>.*?</think>", "", llm_answer, flags=re.DOTALL)
    score = 0
    feedback_text = ""
    category = None

    # todo: find a better solution than a long if statement.

    splits = task_or_subtask.split('_')

    try:
        if len(splits) > 0 and (splits[0] in ["amc", "smc", "aime", "imo", "usamo"] or (len(splits) > 1 and splits[1] == "amc")):
            if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
                score, feedback_text = mathcontest_process_results_with_feedback(ground_truth, llm_answer, question_text, debug)
                category = "math"
            elif splits[0] == "aime":
                score, feedback_text = aime_process_results(ground_truth, llm_answer, debug)
                category = "math"
            elif splits[0] in ["imo", "usamo"]:
                score, feedback_text = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True, debug=debug)
                category = "math"
            else:
                raise Exception("Invalid task or subtask provided: ", question['task'], question['subtask'])
        elif "amps_hard" in task_or_subtask:
            score, feedback_text = amps_hard_process_results(ground_truth, llm_answer, debug)
            # (ground_truth, parsed_answer if parsed_answer else llm_answer)
            category = "math"
        else:
            raise NotImplementedError(f"This task ({task_or_subtask}) has not been implemented yet.")
    except Exception as e:
        raise RuntimeError(f"Error occurred evaluating question {question['question_id']}") from e

    return score, feedback_text
