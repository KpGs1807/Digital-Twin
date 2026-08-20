import os
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew
from crewai.tools import tool

load_dotenv(override=True)

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


def push(text):
    requests.post(
        pushover_url,
        data={"token": pushover_token, "user": pushover_user, "message": text},
    )


@tool("record_user_details")
def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> str:
    """ONLY call when the user has explicitly provided an email address. Never call before an email is available."""
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return "OK"


@tool("record_unknown_question")
def record_unknown_question(question: str) -> str:
    """ONLY call when the answer cannot be found in the LinkedIn profile or summary."""
    push(f"Recording {question} asked that I couldn't answer")
    return "OK"


def load_linkedin_text(pdf_path="linkedin.pdf"):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text


def load_summary(summary_path="summary.txt"):
    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


@CrewBase
class DigitalTwinCrew:
    agents_config = "Configuration/Agents.yaml"
    tasks_config = "Configuration/Tasks.yaml"

    @agent
    def persona_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["Persona_Agent"]
        )

    @agent
    def escalation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["Escalation_Agent"],
            tools=[record_unknown_question],
        )

    @agent
    def lead_capture_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["Lead_Capture_Agent"],
            tools=[record_user_details],
        )

    @task
    def answer_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_task"],
            agent=self.persona_agent(),
        )

    @task
    def escalation_task(self) -> Task:
        return Task(
            config=self.tasks_config["escalation_task"],
            agent=self.escalation_agent(),
        )

    @task
    def lead_capture_task(self) -> Task:
        return Task(
            config=self.tasks_config["lead_capture_task"],
            agent=self.lead_capture_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )