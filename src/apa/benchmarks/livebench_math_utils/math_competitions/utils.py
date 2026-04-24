from typing import Tuple
from ..util import last_boxed_only_string, remove_boxed
import re


def mathcontest_process_results(ground_truth: str, llm_answer: str, question_text: str, debug=False) -> int:
    score = 0
    # the reference answer must be a single capital letter from A to E (I.e., the multiple choice answer)
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and 'A' <= ground_truth <= 'E'):
        raise ValueError("amc_answer must be a single capital letter between A and E.")
    
    # extract text from <solution></solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if len(solution_matches) > 0:
        solution_match = solution_matches[-1]
        if len(set(solution_match)) == 1 and next(iter(set(solution_match))).lower() == ground_truth.lower():
            score = 1

    # The LLM was prompted to repeat letter answer 5 times, to make it easy to pull out the answer        
    if ground_truth * 4 in llm_answer:
        score = 1

    parsed_answer = None

    allow_boxed = True
    if score == 0 and allow_boxed:
        llm_answer = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer)
        if last_boxed:
            last_boxed_res = remove_boxed(last_boxed).replace('\\text{', '').replace('}', '').replace('\\', '').lower()
            if last_boxed_res in {'a', 'b', 'c', 'd', 'e'}:
                parsed_answer = last_boxed_res
            if parsed_answer == ground_truth.lower():
                score = 1

    allow_answer_values = True
    if score == 0 and allow_answer_values:
        value = extract_answer(question_text, ground_truth)
        length_to_check = 20 + len(value)
        if value in llm_answer[-length_to_check:]:
            score = 1

    allow_last_line = True
    if score == 0 and allow_last_line:
        last_line = llm_answer.strip().split('\n')[-1]
        if last_line.strip().replace('*', '').lower() == ground_truth.lower():
            score = 1
        elif '(' in last_line and ')' in last_line:
            val = last_line.split('(')[1].split(')')[0]
            if val.lower() == ground_truth.lower():
                score = 1


    if debug and score == 0:
        # check if the LLM guessed a letter, even if it was wrong
        for letters in ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]:
            if letters in llm_answer:
                parsed_answer = letters[0].lower()

    if debug and score == 0:
        print("INCORRECT")
        print("GROUND TRUTH", ground_truth.strip().lower())
        if parsed_answer:
            print("PARSED ANSWER:", parsed_answer)
        print("END OF OUTPUT", llm_answer[-200:])      

    return score

def mathcontest_process_results_with_feedback(ground_truth: str, llm_answer: str, question_text: str, debug=False):
    score = 0
    feedback_details = []   # Will collect debug info for feedback
    parsed_answer = None

    # the reference answer must be a single capital letter from A to E (i.e. the multiple choice answer)
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and 'A' <= ground_truth <= 'E'):
        raise ValueError("amc_answer must be a single capital letter between A and E.")

    # 1. <solution> tag check
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if len(solution_matches) > 0:
        solution_match = solution_matches[-1]
        if len(set(solution_match)) == 1 and next(iter(set(solution_match))).lower() == ground_truth.lower():
            score = 1
            feedback_details.append(f"Correct answer '{solution_match}' found in <solution> tags (matches ground truth '{ground_truth}').")
        else:
            feedback_details.append(f"Answer between <solution> tags ('{solution_match}') did not match ground truth ('{ground_truth}').")
    else:
        if score != 1:
            feedback_details.append("No <solution> tags found.")

    # 2. 4x repeated letter (e.g., 'CCCC')
    if score == 0:
        if ground_truth * 4 in llm_answer:
            score = 1
            feedback_details.append(f"Correct answer '{ground_truth * 5}' detected as 5x letter repetition.")
        else:
            feedback_details.append("Did not detect 5x repeated letter form of answer. Expected: " + ground_truth * 5)

    # 3. Boxed/fbox (e.g., \boxed{C} or \fbox{C})
    allow_boxed = True
    if score == 0 and allow_boxed:
        llm_answer_boxed = llm_answer.replace("\\\\fbox{", "\\\\boxed{")
        last_boxed = last_boxed_only_string(llm_answer_boxed)
        if last_boxed:
            last_boxed_res = remove_boxed(last_boxed).replace('\\text{', '').replace('}', '').replace('\\', '').lower()
            if last_boxed_res in {'a', 'b', 'c', 'd', 'e'}:
                parsed_answer = last_boxed_res
                feedback_details.append(f"Found boxed answer: '{last_boxed_res.upper()}' (parsed from LaTeX box environment).")
                if parsed_answer == ground_truth.lower():
                    score = 1
                    feedback_details.append(f"Boxed answer matches ground truth '{ground_truth}'.")
                else:
                    feedback_details.append(f"Boxed answer does NOT match ground truth '{ground_truth}'.")
            else:
                feedback_details.append(f"Boxed content '{last_boxed_res}' is not a valid option (A-E).")
        else:
            if score != 1:
                feedback_details.append("No boxed answer found.")

    # 4. Check explicit value at the end (as in "The answer is C.")
    allow_answer_values = True
    if score == 0 and allow_answer_values:
        value = extract_answer(question_text, ground_truth)
        length_to_check = 20 + len(value)
        if value in llm_answer[-length_to_check:]:
            score = 1
            feedback_details.append(f"Found explicit answer value '{value}' at the end of output (matches ground truth '{ground_truth}').")
        else:
            feedback_details.append(f"Did not find explicit answer value '{value}' at the end of the response.")

    # 5. Last line matching (stripped of *, whitespace, etc)
    allow_last_line = True
    if score == 0 and allow_last_line:
        last_line = llm_answer.strip().split('\n')[-1]
        last_line_stripped = last_line.strip().replace('*', '').lower()
        if last_line_stripped == ground_truth.lower():
            score = 1
            feedback_details.append(f"Last line '{last_line_stripped}' matches ground truth '{ground_truth}'.")
        elif '(' in last_line and ')' in last_line:
            val = last_line.split('(')[1].split(')')[0]
            if val.lower() == ground_truth.lower():
                score = 1
                feedback_details.append(f"Last line parenthetical '{val}' matches ground truth '{ground_truth}'.")
            else:
                feedback_details.append(f"Last line parenthetical answer '{val}' does NOT match ground truth '{ground_truth}'.")
        else:
            feedback_details.append(f"Last line '{last_line.strip()}' does not match ground truth '{ground_truth}'.")

    # Compose feedback message
    if score == 1:
        feedback_text = "Correct answer detected.\n" + "\n".join(feedback_details)
    else:
        feedback_text = (
            "Incorrect answer detected.\n"
            "Evaluation details:\n"
            + "\n".join(feedback_details)
            + "\n\n"
            "Tips for improvement:\n"
            "- Explicitly output your final answer as a single letter (A-E), preferably as the last output, in a <solution> tag, as a 5x repeated letter, or boxed in LaTeX.\n"
            "- Avoid any formatting or ambiguity in the answer portion: be direct and clear about your choice."
        )

    return score, feedback_text


def extract_answer(statement, letter):

    pattern = r'\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)'
    matches = re.findall(pattern, statement)
    answers = {match[0]: match[1].strip() for match in matches}
    answer = answers.get(letter, None)

    if not answer or answer == "":
        # this only happens for one question, which is too long for the LLMs to repeat
        answer = "FAILURE"

    answer = answer.strip()
    answer = answer.strip("$")
    answer = answer.strip("~")

    return answer


def aime_process_results(ground_truth: str, llm_answer: str, debug=False):
    score = 0
    if ground_truth in llm_answer[-50:]:
        score = 1
        feedback_text = f"Your answer is correct. The ground truth was '{ground_truth}', and your answer contained the correct value within the last 50 characters."
    else:
        feedback_text = f"Your answer is incorrect. The ground truth was '{ground_truth}'. Your answer did not contain the correct value within the last 50 characters. Note that the correct answer must occur within the last 50 characters of your response to be considered correct."

    if debug and score == 0:
        print('INCORRECT')
        print('GROUND TRUTH', ground_truth)
        print('SOLUTION', llm_answer[-200:])
    return score, feedback_text
