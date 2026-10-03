"""
ReAct Agent with Custom Tools
This project demonstrates a LangGraph ReAct agent using three custom tools:
a calculator, a dictionary/definition lookup and a date/time tool.
The agent uses MemorySaver to maintain conversation memory across turns.
"""

# ============================================================
# 1. IMPORTS AND MODEL SETUP
# ============================================================

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from datetime import datetime
import ast
import operator

# Create the local Ollama model
model = ChatOllama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434"
)

print("Ollama model connected successfully!")

# ============================================================
# 2. CALCULATOR TOOL
# ============================================================

@tool
def calculate(expression: str) -> str:
    """Safely calculate a basic mathematical expression."""

    expression = expression.lower().strip()

    # Convert common English math words to operators
    expression = expression.replace("multiplied by", "*")
    expression = expression.replace("divided by", "/")
    expression = expression.replace("plus", "+")
    expression = expression.replace("minus", "-")

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def evaluate(node):

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            return node.value

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.operand)
            )

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.left),
                evaluate(node.right)
            )

        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)
        return str(result)

    except Exception:
        return "Unable to calculate this expression."
    
# ============================================================
# 3. DICTIONARY / DEFINITION TOOL
# ============================================================

@tool
def define_word(word: str) -> str:
    """Return a simple definition for a word."""

    definitions = {
        "artificial intelligence":
            "The ability of machines to perform tasks that normally require human intelligence.",
        "artificial":
            "Made or produced by humans rather than occurring naturally.",

        "intelligence":
            "The ability to learn, understand, reason and solve problems.",

        "algorithm":
            "A step-by-step procedure used to solve a problem or complete a task.",

        "photosynthesis":
            "The process by which plants use sunlight, water and carbon dioxide to produce food and oxygen.",

        "computer":
            "An electronic device that processes data and performs instructions."
    }

    return definitions.get(
        word.lower(),
        f"No definition found for '{word}'."
    )

# ============================================================
# 4. DATE AND TIME TOOL
# ============================================================

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""

    return datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

# ============================================================
# 5. TEST CUSTOM TOOLS
# ============================================================

print("\nCalculator test:")
print(
    calculate.invoke({
        "expression": "25 * 4 + 10"
    })
)

print("\nDictionary test:")
print(
    define_word.invoke({
        "word": "algorithm"
    })
)

print("\nDate/Time test:")
print(
    get_current_datetime.invoke({})
)

# ============================================================
# 6. CREATE REACT AGENT WITH MEMORY
# ============================================================

tools = [
    calculate,
    define_word,
    get_current_datetime
]

memory = MemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

print("\nReAct agent with memory created successfully!")

# ============================================================
# 7. HELPER FUNCTION TO DISPLAY REACT LOOP
# ============================================================

def run_agent(question, config):
    """
    Run the agent and display the ReAct loop:
    Think -> Act -> Observe -> Answer
    """

    response = agent.invoke(
        {
            "messages": [
                ("user", question)
            ]
        },
        config=config
    )

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("REACT LOOP")
    print("=" * 60)

    for message in response["messages"]:

        # Tool call made by the agent
        if hasattr(message, "tool_calls") and message.tool_calls:

            print("\nACTION:")
            for tool_call in message.tool_calls:
                print(
                    f"Tool: {tool_call['name']}"
                )
                print(
                    f"Input: {tool_call['args']}"
                )

        # Tool observation
        elif type(message).__name__ == "ToolMessage":

            print("\nOBSERVATION:")
            print(message.content)

        # Final AI answer
        elif type(message).__name__ == "AIMessage":

            if message.content:
                print("\nANSWER:")
                print(message.content)

    return response

# ============================================================
# 8. INITIAL REACT TEST
# ============================================================

initial_config = {
    "configurable": {
        "thread_id": "initial-react-test"
    }
}

run_agent(
    "Use the calculator tool to calculate 25 * 16.",
    initial_config
)

# ============================================================
# 9. FIVE REQUIRED TEST QUESTIONS
# ============================================================

test_questions = [

    # Tool question 1 - Calculator
    "Use the calculator tool to calculate 125 divided by 5.",

    # Tool question 2 - Dictionary
    "Use the dictionary tool to define the word algorithm.",

    # Direct question 1
    "What is artificial intelligence?",

    # Direct question 2
    "Why is Python popular for programming?",

    # Tool + reasoning question
    "Use the calculator tool to first calculate 3 * 12, then subtract 5 from the result. How many apples are left?"
]

for i, question in enumerate(test_questions, start=1):

    print("\n" + "#" * 60)
    print(f"QUESTION {i} OF 5")
    print("#" * 60)

    # Give every independent test its own conversation thread
    test_config = {
        "configurable": {
            "thread_id": f"test-question-{i}"
        }
    }

    run_agent(
        question,
        test_config
    )

# ============================================================
# 10. TEST CONVERSATION MEMORY
# ============================================================

memory_config = {
    "configurable": {
        "thread_id": "memory-demo"
    }
}

# First turn
first_question = "My name is Antra."

first_response = agent.invoke(
    {
        "messages": [
            ("user", first_question)
        ]
    },
    config=memory_config
)

print("\n" + "=" * 60)
print("MEMORY TEST - TURN 1")
print("=" * 60)
print(first_response["messages"][-1].content)

# Second turn - same thread
second_question = "What is my name?"

second_response = agent.invoke(
    {
        "messages": [
            ("user", second_question)
        ]
    },
    config=memory_config
)

print("\n" + "=" * 60)
print("MEMORY TEST - TURN 2")
print("=" * 60)
print(second_response["messages"][-1].content)

# ============================================================
# 11. DATE/TIME TOOL TEST
# ============================================================

datetime_config = {
    "configurable": {
        "thread_id": "datetime-demo"
    }
}

datetime_question = "What is the current date and time?"

datetime_response = agent.invoke(
    {
        "messages": [
            ("user", datetime_question)
        ]
    },
    config=datetime_config
)

print("\n" + "=" * 60)
print("DATE/TIME TOOL TEST")
print("=" * 60)

print(datetime_response["messages"][-1].content)

# ============================================================
# 12. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("REACT AGENT TESTING COMPLETED SUCCESSFULLY")
print("=" * 60)
