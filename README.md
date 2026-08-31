# GROWAI LLM Engineering – Assignment 7

## ReAct Agent with Custom Tools

This project demonstrates a ReAct-based AI agent built with LangChain and a local Qwen3 0.6B model using Ollama.

## Purpose

The purpose of this assignment is to understand how an AI agent can dynamically decide whether to use a tool or answer a question directly. The project demonstrates the ReAct workflow, custom tool integration and conversation memory.

## Features

- Builds a ReAct-based AI agent using LangChain
- Uses a local Qwen3 0.6B model through Ollama
- Implements a custom Calculator tool
- Implements a Dictionary / Definition lookup tool
- Implements a Date and Time tool
- Demonstrates dynamic tool selection by the agent
- Demonstrates the ReAct workflow: Think → Act → Observe → Answer
- Uses MemorySaver for conversation memory
- Tests tool-based and direct questions
- Includes a question requiring tool output with reasoning
- Demonstrates conversation memory across multiple turns
- Includes live execution and output demonstration

## Requirements

- Python 3.x
- Ollama
- Qwen3 0.6B
- LangChain
- LangGraph

Install the required dependencies using:
```text
pip install -r requirements.txt
```

Make sure Ollama is installed and the Qwen3 0.6B model is available locally:
```text
ollama pull qwen3:0.6b
```

## Setup / Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies using `requirements.txt`.
4. Make sure Ollama is installed and running.
5. Make sure the `qwen3:0.6b` model is available locally.

## How to Run

Run the Python script using:
```text
python agents.py
```

The program initializes the local LLM, creates the custom tools, builds the ReAct agent, processes different questions, demonstrates tool usage and tests conversation memory.

## Custom Tools

### Calculator Tool

The Calculator tool safely evaluates basic mathematical expressions using Python's `ast` module and a restricted set of mathematical operators.

### Dictionary / Definition Tool

The Dictionary tool provides simple definitions for predefined words from a local dictionary.

### Date and Time Tool

The Date and Time tool returns the current local date and time using Python's `datetime` module.

## ReAct Workflow

The agent follows the ReAct pattern:

### Think → Act → Observe → Answer

The agent analyzes the user's request, decides whether a tool is required, executes the selected tool, observes its result and then generates the final response.

## Test Cases

The project tests different types of questions:

1. A mathematical question requiring the Calculator tool.
2. A definition question requiring the Dictionary tool.
3. A general question that can be answered directly.
4. Another general knowledge question.
5. A question requiring tool output combined with reasoning.

The project also includes a separate conversation-memory test using two turns within the same session.

## Conversation Memory

`MemorySaver` is used as the agent checkpointer to maintain conversation state within a session.

For example:

`Turn 1`: The user provides their name.

`Turn 2`: The user asks the agent to recall their name.

This demonstrates that the agent can use information from the previous turn.

## Project Files

- `agents.py` – Main Python implementation containing the Ollama model, custom tools, ReAct agent, memory and test cases.
- `requirements.txt` – Required Python dependencies.
- `.gitignore` - Files and folders excluded from Git tracking.

## Real-World Relevance

Tool-using AI agents are useful when an LLM needs capabilities beyond generating text. Similar architectures can be used in AI assistants, customer-support systems, productivity tools, calculation systems, knowledge assistants and applications that interact with external services.

## Edge Case / Failure Point

One potential failure point is invalid or unsupported tool input. The Calculator tool handles unsupported mathematical expressions safely by returning an error message instead of executing arbitrary operations.

Similarly, the Dictionary tool returns a clear message when a requested word is not available in the local dictionary.

## Assignment

GROWAI LLM Engineering & Generative AI – Assignment 7
