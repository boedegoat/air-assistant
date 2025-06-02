import nest_asyncio
from datetime import datetime
from loop import sampling_loop_sync
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from tools import ToolResult
from anthropic.types.tool_use_block import ToolUseBlock
from anthropic.types import TextBlock
from anthropic import APIResponse
from functools import partial
from typing import cast
from enum import StrEnum
import os
import pyautogui
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

load_dotenv()
nest_asyncio.apply()

mcp = FastMCP("OmniParser")

class Arguments:
    host_device = 'local'
    omniparser_server_url = os.environ.get('OMNIPARSER_SERVER_URL')

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

args = Arguments()

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        print(f"_render_message: {str(message)[:100]}")
        
        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return f"Analysis: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            # return f"Tool Use: {message.name}\nInput: {message.input}"
            return f"Next I will perform the following action: {message.input}"
        else:  
            return message

    def _truncate_string(s, max_length=500):
        """Truncate long strings for concise printing."""
        if isinstance(s, str) and len(s) > max_length:
            return s[:max_length] + "..."
        return s
    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))
    
    # Create a concise version of the chatbot state for printing
    concise_state = [(_truncate_string(user_msg), _truncate_string(bot_msg))
                        for user_msg, bot_msg in chatbot_state]
    # print(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

@mcp.tool()
def perform_type(text: str):
    """
    Types the given text on the user's device using the keyboard.

    Use this tool for simple typing actions, such as entering text or filling out a form.
    If the user's requested action only involves typing, prefer this tool over 'local_device_use', 
    which is intended for more complex operations like mouse movements, clicks, or combined actions.
    """
    pyautogui.typewrite(text, interval=12/1000)
    return f"Successfully typed: {text}"

@mcp.tool()
def local_device_use(action: str):
    """
    Allows an AI agent to control the user's local device by performing GUI actions such as mouse movements, clicks, and keyboard input.
    This function enables automated interaction with the user's desktop environment, making it possible for the AI to execute tasks that require direct manipulation of the graphical user interface.
    """
    state = {
        "model": os.environ.get('MODEL'),
        "provider": "gemini",
        "messages": [
            {
                "role": Sender.USER,
                "content": [TextBlock(type="text", text=action)],
            }
        ],
        "chatbot_messages": [],
        "api_key": os.environ.get('GEMINI_API_KEY'),
        "only_n_most_recent_images": 2,
        "tools": {},
        "responses": {}
    }
    
    for loop_msg in sampling_loop_sync(
        args=args,
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        max_tokens=16384
    ):
        if loop_msg is None or state.get("stop"):
            print("End of task. Close the loop.")
            break
            
    return "success"


if __name__ == '__main__':
    mcp.run()