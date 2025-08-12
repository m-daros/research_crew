from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from panel.chat import ChatInterface

from research_crew.tools.tools import search_tool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
chat_interface = ChatInterface ()

def print_output ( output: TaskOutput ):
    message = output.raw
    chat_interface.send ( message, user = output.agent, respond = False )


@CrewBase
class ResearchCrew ():
    """ResearchCrew crew"""

    agents: List [ BaseAgent ]
    tasks: List [ Task ]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def research_planner ( self ) -> Agent:
        return Agent (
                config = self.agents_config [ "research_planner" ],  # type: ignore[index]
                verbose = True
        )

    @agent
    def research_assistant ( self ) -> Agent:
        return Agent (
                config = self.agents_config [ "research_assistant" ],  # type: ignore[index]
                verbose = True
        )


    @agent
    def writer ( self ) -> Agent:
        return Agent (
                config = self.agents_config [ "writer" ],  # type: ignore[index]
                verbose = True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def plan_task ( self ) -> Task:
        return Task (
                config = self.tasks_config [ "plan_task" ],  # type: ignore[index]
                callback = print_output
        )

    @task
    def research_task ( self ) -> Task:
        return Task (
                config = self.tasks_config [ "research_task" ],  # type: ignore[index]
                tools = [ search_tool ],
                callback = print_output
        )

    @task
    def writing_task ( self ) -> Task:
        return Task (
                config = self.tasks_config [ "writing_task" ],  # type: ignore[index]
                callback = print_output,
                human_input = True
        )

    @crew
    def crew ( self ) -> Crew:
        """Creates the ResearchCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew (
                agents = self.agents,  # Automatically created by the @agent decorator
                tasks = self.tasks,  # Automatically created by the @task decorator
                process = Process.sequential,
                verbose = True,
                # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
