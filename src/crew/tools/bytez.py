from bytez import Bytez
from crewai.tools import tool
sdk = Bytez("008500fe8f2fd1acbdd6ae89df607343")
# choose medical-chat-v0
model = sdk.model("moriire/medical-chat-v0")

# send input to model
@tool("bytez medical model tool")
def run_model(input_text: str) -> dict:
    """
    Run the model with the given input text user query.
    
    Args:
        input_text (str): The input text to send to the model.
        
    Returns:
        dict: The output from the model, including any error messages.
    """
    output, error = model.run([
        {
            "role": "user",
            "content": input_text
        }
    ])
    
    return {"error": error, "output": output}