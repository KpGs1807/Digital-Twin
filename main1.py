from dotenv import load_dotenv
load_dotenv(override=True)

from crew1 import DigitalTwinCrew, load_summary, load_linkedin_text


def run_twin(message, history):
    inputs = {
        "message": message,
        "history": str(history),
        "summary": load_summary(),
        "linkedin": load_linkedin_text(),
    }
    result = DigitalTwinCrew().crew().kickoff(inputs=inputs)

    # tasks_output[0] = answer_task's output (the actual reply the visitor should see).
    # result.raw would instead give the LAST task's output (lead_capture_task),
    # which is just a confirmation/no-op message, not the answer.
    return result.tasks_output[0].raw


if __name__ == "__main__":
    reply = run_twin("I want to contact prajwal this is my email:abtesty.com", [])
    print(reply)