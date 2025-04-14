import requests
import json
import os
import time


class MistralClient:
    """
    A direct client for the Mistral API without any dependency on CrewAI or LiteLLM.
    """

    def __init__(self, api_url):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def generate_response(self, prompt, max_tokens=1000, temperature=0.3):
        """
        Generate a response from the Mistral API using the exact format required.
        """
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "parameters": {
                        "extra": {
                            "temperature": temperature,
                            "max_new_tokens": max_tokens,
                            "repetition_penalty": 1
                        }
                    },
                    "inputs": [
                        {
                            "name": "input",
                            "shape": [1],
                            "datatype": "str",
                            "data": [prompt]
                        }
                    ]
                },
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

            if 'outputs' in result and len(result['outputs']) > 0:
                return result['outputs'][0]['data'][0]
            else:
                raise ValueError(f"Unexpected response format: {result}")

        except Exception as e:
            print(f"Error calling Mistral API: {str(e)}")
            raise


class AgentRunner:
    """
    A lightweight agent runner that directly uses the Mistral API
    without relying on CrewAI's internal LLM handling.
    """

    def __init__(self, api_url):
        self.client = MistralClient(api_url)

    def run_task(self, agent_role, agent_goal, task_description, agent_backstory=""):
        """
        Run a single task with a specific agent role and goal.

        Args:
            agent_role: The role of the agent (e.g., "Data Scientist")
            agent_goal: The goal of the agent (e.g., "Analyze data patterns")
            task_description: The description of the task to perform
            agent_backstory: Background context for the agent

        Returns:
            The response from the LLM as a string
        """
        prompt = self.format_agent_prompt(agent_role, agent_goal, task_description, agent_backstory)
        return self.client.generate_response(prompt)

    def format_agent_prompt(self, agent_role, agent_goal, task_description, agent_backstory=""):
        """
        Format a prompt for an agent task in a way that resembles CrewAI's format.
        """
        return f"""<s>[INST] System: You are {agent_role}.

Your goal is: {agent_goal}

{agent_backstory if agent_backstory else ''}

You need to accomplish this task:
{task_description}

Please be thorough in your work and show your reasoning step-by-step.
[/INST]</s>"""

    def run_workflow(self, workflow_tasks):
        """
        Run a sequence of tasks with different agent roles.

        Args:
            workflow_tasks: A list of dictionaries, each containing:
                - role: The agent role
                - goal: The agent goal
                - task: The task description
                - backstory (optional): The agent backstory

        Returns:
            A dictionary mapping task indices to responses
        """
        results = {}
        context = ""

        for i, task_info in enumerate(workflow_tasks):
            print(f"\n{'=' * 80}\nRunning task {i + 1}/{len(workflow_tasks)}: {task_info['role']}\n{'=' * 80}")

            # Add context from previous tasks if available
            task_with_context = task_info['task']
            if context:
                task_with_context += f"\n\nContext from previous tasks:\n{context}"

            # Run the task
            start_time = time.time()
            result = self.run_task(
                task_info['role'],
                task_info['goal'],
                task_with_context,
                task_info.get('backstory', '')
            )
            end_time = time.time()

            # Store result
            results[i] = result

            # Add to context for next task
            context += f"\n\nTask {i + 1} ({task_info['role']}) result:\n{result}\n"

            print(f"Completed in {end_time - start_time:.2f} seconds")
            print(f"Result preview: {result[:200]}...")

        return results


# Example usage
def main():
    api_url = "https://aiplatform.dev51.cbf.dev.paypalinc.com/seldon/seldon/mistral-7b-inst-624b0/v2/models/mistral-7b-inst-624b0/infer"
    runner = AgentRunner(api_url)

    # Define a simple workflow
    workflow = [
        {
            "role": "Data Understanding Specialist",
            "goal": "Analyze data and validate configuration compatibility",
            "task": "Given a dataset with columns [customer_id, transaction_amount, timestamp, is_fraud], describe the key data patterns we should look for in fraud detection.",
            "backstory": "You are an expert in financial data analysis with specialization in fraud detection."
        },
        {
            "role": "Feature Engineering Specialist",
            "goal": "Create optimal features for fraud detection",
            "task": "Based on the data understanding, suggest 5 features we should extract from this dataset for optimal fraud detection."
        }
    ]

    results = runner.run_workflow(workflow)

    # Print all results
    for i, result in results.items():
        print(f"\nTask {i + 1} Result:")
        print("-" * 40)
        print(result)
        print("-" * 40)


if __name__ == "__main__":
    main()