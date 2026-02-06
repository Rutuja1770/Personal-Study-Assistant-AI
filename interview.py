from assistant import ask_assistant

def interview_response(answer):
    prompt = f"""
You are a technical interviewer.
Ask follow-up questions and give feedback.

Candidate Answer:
{answer}
"""
    return ask_assistant(prompt)
