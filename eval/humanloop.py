# https://humanloop.com/docs/quickstart/evals-in-code#install-and-initialize-the-sdk
from humanloop import Humanloop

# Get API key at https://app.humanloop.com/account/api-keys
hl = Humanloop(api_key="<YOUR Humanloop API KEY>")

checks = hl.evaluations.run(
    name="Initial Test",
    file={
        "path": "Scifi/App",
        "callable": lambda messages: (
            "I'm sorry, Dave. I'm afraid I can't do that."
            if messages[-1]["content"].lower() == "hal"
            else "Beep boop!"
        )  # replace the lambda with your AI model
    },
    dataset={
        "path": "Scifi/Tests",
        "datapoints": [
            {
                "messages": [
                    {"role": "system", "content": "You are an AI that responds like famous sci-fi AIs."},
                    {"role": "user", "content": "HAL"}
                ],
                "target": {"output": "I'm sorry, Dave. I'm afraid I can't do that."}
            },
            {
                "messages": [
                    {"role": "system", "content": "You are an AI that responds like famous sci-fi AIs."},
                    {"role": "user", "content": "R2D2"}
                ],
                "target": {"output": "Beep boop beep!"}
            }
        ]
    }, #  replace with your test-dataset
    evaluators=[
        {"path": "Example Evaluators/Code/Exact match"},
        {"path": "Example Evaluators/Code/Latency"},
        {"path": "Example Evaluators/AI/Semantic similarity"},
    ],
)

