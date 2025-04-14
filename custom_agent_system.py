import requests
import json
import time


class CustomLLMClient:
    """A simple client for your custom LLM API with enhanced logging"""

    def __init__(self, api_url):
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json"
        }

    def generate_response(self, prompt, system_prompt=None):
        """Generate a response from the custom API with detailed logging"""
        # Log the prompt
        print("\n" + "=" * 80)
        print("SENDING PROMPT TO LLM:")
        print("-" * 50)
        print(f"SYSTEM: {system_prompt}")
        print("-" * 50)
        print(f"USER: {prompt}...")  # First 500 chars to avoid too much output
        print("=" * 80 + "\n")

        inputs = [
            {
                "name": "input",
                "shape": [1],
                "datatype": "str",
                "data": [prompt]
            }
        ]

        # Add system prompt if provided
        if system_prompt:
            inputs.append({
                "name": "system",
                "shape": [1],
                "datatype": "str",
                "data": [system_prompt]
            })

        payload = {
            "parameters": {
                "extra": {
                    "temperature": 0.3,
                    "max_new_tokens": 512,  # Increased token limit
                    "repetition_penalty": 1
                }
            },
            "inputs": inputs
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            result = response.json()

            # Extract the response text
            if "outputs" in result and len(result["outputs"]) > 0:
                response_text = result["outputs"][0]["data"][0]

                # Log the response
                print("\n" + "=" * 80)
                print("LLM RESPONSE:")
                print("-" * 50)
                print(response_text)
                print("=" * 80 + "\n")

                return response_text
            else:
                error_msg = "Error: Unexpected response format from API"
                print(f"\nLLM ERROR: {error_msg}\n")
                return error_msg
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            print(f"\nLLM ERROR: {error_msg}\n")
            return error_msg


class SimpleAgent:
    """A simplified agent implementation that works with custom LLM API"""

    def __init__(self, role, goal, backstory, llm_client):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm_client = llm_client
        self.tools = []

    def add_tool(self, tool):
        """Add a tool to this agent"""
        self.tools.append(tool)

    def execute_task(self, task, context=None):
        """Execute a task with proper tool input parsing"""
        print(f"\n{'=' * 80}\nAgent {self.role} executing task\n{'=' * 80}")

        # Create the system prompt
        system_prompt = f"You are {self.role}. Your goal is: {self.goal}. {self.backstory}"

        # Create the task prompt including context
        task_prompt = f"TASK: {task}\n\n"

        if context:
            task_prompt += f"CONTEXT FROM PREVIOUS TASKS:\n{context}\n\n"

        # Add information about available tools
        if self.tools:
            task_prompt += "AVAILABLE TOOLS:\n"
            for tool in self.tools:
                task_prompt += f"- {tool['name']}: {tool['description']}\n"

            task_prompt += "\nTo use a tool, format your response like this:\n"
            task_prompt += "THINKING: your reasoning about what to do\n"
            task_prompt += "TOOL: tool_name\n"
            task_prompt += "TOOL_INPUT: {}\n"  # Empty tool input by default
            task_prompt += "\nAfter using a tool, provide your final answer:\n"
            task_prompt += "FINAL ANSWER: your final response to the task\n"
        else:
            task_prompt += "\nProvide your response directly."

        # Generate the initial response
        response = self.llm_client.generate_response(task_prompt, system_prompt)

        # Check if the response contains a tool call
        if "TOOL:" in response:
            try:
                # Extract tool name and input
                thinking = response.split("THINKING:")[1].split("TOOL:")[0].strip() if "THINKING:" in response else ""
                tool_part = response.split("TOOL:")[1].split("\n")[0].strip()
                tool_input_part = response.split("TOOL_INPUT:")[1].split("\n")[
                    0].strip() if "TOOL_INPUT:" in response else "{}"

                print(f"Agent is thinking: {thinking}")
                print(f"Agent wants to use tool: {tool_part}")

                # Normalize tool name for comparison (remove spaces, lowercase)
                normalized_tool_name = tool_part.lower().replace(" ", "_").strip()

                # Find the tool with more flexible matching
                tool = None
                for t in self.tools:
                    if t['name'].lower().replace(" ", "_").strip() == normalized_tool_name:
                        tool = t
                        break

                if tool:
                    # Execute the tool
                    print(f"Executing tool: {tool['name']}")
                    try:
                        # IMPORTANT FIX: Don't try to parse the input at all, just call with no arguments
                        # Most of our tools don't need input parameters
                        tool_result = tool['func']()

                        # Create follow-up prompt with tool result
                        followup_prompt = task_prompt + "\n\n" + response + "\n\n"
                        followup_prompt += f"TOOL RESULT:\n{tool_result}\n\n"
                        followup_prompt += "Based on this result, provide your final answer."

                        # Get the final response
                        final_response = self.llm_client.generate_response(followup_prompt, system_prompt)

                        if "FINAL ANSWER:" in final_response:
                            final_answer = final_response.split("FINAL ANSWER:")[1].strip()
                        else:
                            final_answer = final_response

                        return {
                            "thinking": thinking,
                            "tool_used": tool['name'],
                            "tool_result": tool_result,
                            "response": final_answer
                        }
                    except Exception as e:
                        print(f"Error executing tool {tool['name']}: {str(e)}")
                        error_message = f"I encountered an error while using the {tool['name']} tool: {str(e)}. "
                        error_message += "Let me try to provide a helpful response without using the tool."

                        # Ask the LLM to recover from the error
                        recovery_prompt = task_prompt + "\n\n" + response + "\n\n"
                        recovery_prompt += f"ERROR: {str(e)}\n\n"
                        recovery_prompt += ("The tool you tried to use encountered an error. Please provide your best "
                                            "response without using the tool.")

                        recovery_response = self.llm_client.generate_response(recovery_prompt, system_prompt)

                        return {
                            "thinking": thinking,
                            "tool_error": str(e),
                            "response": recovery_response
                        }
                else:
                    print(f"Tool not found: {tool_part}")

                    # Ask the LLM to recover when tool not found
                    recovery_prompt = task_prompt + "\n\n" + response + "\n\n"
                    recovery_prompt += f"ERROR: Tool '{tool_part}' not found. Available tools: {[t['name'] for t in self.tools]}\n\n"
                    recovery_prompt += ("The tool you tried to use was not found. Please provide your best response "
                                        "using one of the available tools or without using a tool.")

                    recovery_response = self.llm_client.generate_response(recovery_prompt, system_prompt)

                    # Check if the recovery response tries to use a tool
                    if "TOOL:" in recovery_response:
                        # Process the new tool attempt recursively
                        return self.execute_task(task, context + "\n\nPrevious attempt: " + response)

                    return {
                        "response": f"I wanted to use the '{tool_part}' tool but couldn't find it. Here's my analysis "
                                    f"without the tool:\n\n{recovery_response}"
                    }
            except Exception as e:
                print(f"Error executing tool: {str(e)}")
                return {
                    "response": f"Error while using tool: {str(e)}. Here's my original response:\n{response}"
                }

        # If no tool was used, return the response directly
        return {
            "response": response
        }


class SimpleWorkflow:
    """A simplified workflow implementation that works with custom agents"""

    def __init__(self, agents, experiment):
        self.agents = agents
        self.experiment = experiment

    def run_workflow(self, tasks):
        """Run the workflow with the given tasks with detailed progress logging"""
        results = []
        context = ""

        print(f"\n\n{'#' * 100}")
        print(f"{'#' * 40} STARTING WORKFLOW EXECUTION {'#' * 40}")
        print(f"{'#' * 100}\n")

        for i, task in enumerate(tasks):
            print(f"\n\n{'#' * 100}")
            print(f"TASK {i + 1}/{len(tasks)}: {task['description'][:100]}...")
            print(f"{'#' * 100}")

            start_time = time.time()

            # Get the agent for this task
            agent = self.agents[task['agent_index']]

            # Update experiment status
            self.experiment.update_status(
                "running",
                (i + 1) / len(tasks),
                f"task_{i + 1}",
                agent.role
            )

            self.experiment.add_log_entry(
                f"Starting task {i + 1}/{len(tasks)}: {agent.role}",
                level="INFO",
                agent=agent.role,
                phase=f"task_{i + 1}"
            )

            # Execute the task
            task_result = agent.execute_task(task['description'], context)

            # Log execution time
            execution_time = time.time() - start_time
            print(f"\nTask execution time: {execution_time:.2f} seconds")

            # Add to results
            result = {
                "task_id": i + 1,
                "task_description": task['description'],
                "agent_role": agent.role,
                "execution_time_seconds": execution_time,
                "result": task_result
            }
            results.append(result)

            # Log result summary
            if "tool_used" in task_result:
                result_summary = f"Used tool: {task_result['tool_used']}"
            elif "tool_error" in task_result:
                result_summary = f"Tool error: {task_result['tool_error']}"
            elif "tool_missing" in task_result:
                result_summary = f"Missing tool: {task_result['tool_missing']}"
            elif "direct_response" in task_result and task_result["direct_response"]:
                result_summary = "Direct response (no tool used)"
            else:
                result_summary = "Completed with unspecified path"

            print(f"\n{'=' * 50}")
            print(f"TASK {i + 1} COMPLETED: {result_summary}")
            print(f"{'=' * 50}\n")

            # Log to experiment
            self.experiment.add_log_entry(
                f"Completed task {i + 1}/{len(tasks)}: {result_summary}",
                level="INFO",
                agent=agent.role,
                phase=f"task_{i + 1}"
            )

            # Update context for the next task - truncate if too long
            response_text = task_result.get('response', '')
            if len(response_text) > 2000:
                response_text = response_text[:1000] + "\n...[truncated]...\n" + response_text[-1000:]

            context += f"\n\nTASK {i + 1} ({agent.role}):\n{task['description']}\n\nRESULT:\n{response_text}\n"

            # Sleep briefly to let the API breathe
            time.sleep(1)

        print(f"\n\n{'#' * 100}")
        print(f"{'#' * 40} WORKFLOW EXECUTION COMPLETED {'#' * 40}")
        print(f"{'#' * 100}\n")

        # Log final summary to experiment
        self.experiment.add_log_entry(
            f"Workflow completed with {len(tasks)} tasks",
            level="INFO"
        )

        return results
