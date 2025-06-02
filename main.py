import os
import nest_asyncio
import asyncio
import base64
import io
import traceback
from dotenv import load_dotenv
import json

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types

from mcp_handler import MCPClient

nest_asyncio.apply()
load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
# MODEL = "models/gemini-2.5-flash-preview-native-audio-dialog"

DEFAULT_MODE = "screen"

client = genai.Client(
    http_options={"api_version": "v1beta", "timeout": 999},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.mcp_client = MCPClient()

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def handle_server_content(self, server_content):
        model_turn = server_content.model_turn
        if model_turn:
            for part in model_turn.parts:
                executable_code = part.executable_code
                if executable_code is not None:
                    print('-------------------------------')
                    print(f'``` python\n{executable_code.code}\n```')
                    print('-------------------------------')

                code_execution_result = part.code_execution_result
                if code_execution_result is not None:
                    print('-------------------------------')
                    print(f'```\n{code_execution_result.output}\n```')
                    print('-------------------------------')

        grounding_metadata = getattr(server_content, 'grounding_metadata', None)
        if grounding_metadata is not None:
            print(grounding_metadata.search_entry_point.rendered_content)
    
    async def handle_tool_call(self, tool_call):
        # This is a simplistic approach. If tool names are unique across servers,
        # this will work. If not, or if specific servers are preferred for
        # specific tools, this logic needs to be more sophisticated.
        # For example, you might need to know which server a tool belongs to.
        target_session = None
        if self.mcp_client.sessions:
            # Attempt to find a session that has the tool
            for session_name, session_obj in self.mcp_client.sessions.items():
                try:
                    # Check if the tool is available in this session
                    available_tools_response = await session_obj.list_tools()
                    if any(tool.name == tool_call.function_calls[0].name for tool in available_tools_response.tools):
                        target_session = session_obj
                        print(f"Routing tool call for '{tool_call.function_calls[0].name}' to server: {session_name}")
                        break
                except Exception as e:
                    print(f"Error listing tools for session {session_name}: {e}")
            
            if not target_session: # Fallback to the first session if tool not found or error occurs
                target_session = next(iter(self.mcp_client.sessions.values()), None)
                if target_session:
                    print(f"Warning: Tool '{tool_call.function_calls[0].name}' not explicitly found on any server. Attempting with first available session.")
                else:
                    print(f"Error: No MCP sessions available to handle tool call for '{tool_call.function_calls[0].name}'.")
                    return


        if not target_session:
            print(f"Error: No active MCP session to handle tool call for '{tool_call.function_calls[0].name}'.")
            # Potentially send an error response back to the model.
            # For now, just returning.
            return

        for fc in tool_call.function_calls:
            try:
                result = await target_session.call_tool(
                    name=fc.name,
                    arguments=fc.args,
                )
                print(result)
                tool_response = types.LiveClientToolResponse(
                    function_responses=[types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={'result':result},
                    )]
                )

                print('\n>>> ', tool_response)
                await self.session.send(input=tool_response) # self.session here is the Gemini session
            except Exception as e:
                print(f"Error calling tool {fc.name} on target session: {e}")
                # Send error back to Gemini
                error_response_content = {"error": f"Failed to execute tool {fc.name}: {str(e)}"}
                tool_response = types.LiveClientToolResponse(
                    function_responses=[types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={'result': error_response_content}, # Send error as part of the result
                    )]
                )
                await self.session.send(input=tool_response)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                
                if server_content := response.server_content:
                    self.handle_server_content(server_content)
                    continue

                if tool_call := response.tool_call:
                    await self.handle_tool_call(tool_call)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        stdio_configs = None
        try:
            with open("servers_config.json", "r") as f:
                config_data = json.load(f)
                stdio_configs = config_data.get("mcpServers")
        except FileNotFoundError:
            print("servers_config.json not found. Skipping STDIO server connections.")
        except json.JSONDecodeError:
            print("Error decoding servers_config.json. Skipping STDIO server connections.")

        await self.mcp_client.connect_to_servers(stdio_configs=stdio_configs, connect_sse=True) # Connect to all configured servers

        all_mcp_tools = []
        if self.mcp_client.sessions:
            for server_name, session_obj in self.mcp_client.sessions.items():
                try:
                    response = await session_obj.list_tools()
                    if hasattr(response, 'tools') and response.tools:
                        print(f"Tools from {server_name}: {[tool.name for tool in response.tools]}")
                        all_mcp_tools.extend(response.tools)
                    else:
                        print(f"No tools found for MCP server '{server_name}' or 'response.tools' is not iterable.")
                except Exception as e:
                    print(f"Error listing tools for MCP server '{server_name}': {e}")
        else:
            print("No MCP client sessions established.")
        
        functional_tools = []
        
        if all_mcp_tools:
            # Deduplicate tools by name, preferring the first one encountered
            # This is a simple deduplication. More sophisticated logic might be needed
            # if tools with the same name but different schemas/servers exist.
            seen_tool_names = set()
            unique_mcp_tools = []
            for tool_mcp in all_mcp_tools:
                if tool_mcp.name not in seen_tool_names:
                    unique_mcp_tools.append(tool_mcp)
                    seen_tool_names.add(tool_mcp.name)
            
            print(f"Unique MCP tools available to Gemini: {seen_tool_names}")

            for tool_mcp in unique_mcp_tools:
                gemini_tool_params_schema = None 
                mcp_input_schema = getattr(tool_mcp, 'inputSchema', None)

                tool_name_mcp = getattr(tool_mcp, 'name', 'unknown_tool')
                tool_description_mcp = getattr(tool_mcp, 'description', f'Tool {tool_name_mcp}')

                if mcp_input_schema and isinstance(mcp_input_schema, dict):
                    schema_type_mcp = mcp_input_schema.get("type", "OBJECT").upper()
                    
                    gemini_tool_params_schema = {"type": schema_type_mcp}
                    if "description" in mcp_input_schema:
                        gemini_tool_params_schema["description"] = mcp_input_schema["description"]

                    if schema_type_mcp == "OBJECT":
                        gemini_tool_params_schema["properties"] = {}
                        mcp_properties = mcp_input_schema.get("properties")

                        if isinstance(mcp_properties, dict):
                            for param_name, param_details_mcp in mcp_properties.items():
                                if not isinstance(param_details_mcp, dict):
                                    print(f"Warning (Tool: {tool_name_mcp}): Parameter '{param_name}' has invalid schema (not a dict). Skipping.")
                                    continue

                                param_type_mcp = param_details_mcp.get("type", "").upper()
                                # Ensure param_type_mcp is a string before calling upper()
                                if not isinstance(param_type_mcp, str):
                                    print(f"Warning (Tool: {tool_name_mcp}): Parameter '{param_name}' type is not a string. Skipping.")
                                    continue

                                if not param_type_mcp:
                                    print(f"Warning (Tool: {tool_name_mcp}): Parameter '{param_name}' missing 'type'. Skipping.")
                                    continue
                                
                                current_param_gemini_schema = {
                                    "type": param_type_mcp,
                                    "description": param_details_mcp.get("description", "")
                                }

                                if param_type_mcp == "ARRAY":
                                    mcp_items_schema = param_details_mcp.get("items")
                                    if isinstance(mcp_items_schema, dict):
                                        item_type_mcp = mcp_items_schema.get("type", "").upper()
                                        # Ensure item_type_mcp is a string
                                        if not isinstance(item_type_mcp, str):
                                            print(f"Warning (Tool: {tool_name_mcp}, Param: {param_name}): Array 'items' type is not a string. Defaulting to STRING.")
                                            item_type_mcp = "STRING"
                                        
                                        if item_type_mcp:
                                            gemini_item_details_for_array = {"type": item_type_mcp}
                                            if "description" in mcp_items_schema:
                                                gemini_item_details_for_array["description"] = mcp_items_schema["description"]
                                            current_param_gemini_schema["items"] = gemini_item_details_for_array
                                        else:
                                            print(f"Warning (Tool: {tool_name_mcp}, Param: {param_name}): Array parameter missing 'items' schema or 'items' is not a dict. Defaulting item type to STRING.")
                                            current_param_gemini_schema["items"] = {"type": "STRING"} 
                                    else:
                                        print(f"Warning (Tool: {tool_name_mcp}, Param: {param_name}): Array parameter missing 'items' schema or 'items' is not a dict. Defaulting item type to STRING.")
                                        current_param_gemini_schema["items"] = {"type": "STRING"} 

                                if param_type_mcp == "OBJECT":
                                    nested_mcp_props = param_details_mcp.get("properties")
                                    if isinstance(nested_mcp_props, dict):
                                        current_param_gemini_schema["properties"] = {
                                            np_name: {"type": (np_details.get("type","STRING") if isinstance(np_details.get("type"), str) else "STRING").upper(), "description": np_details.get("description","")}
                                            for np_name, np_details in nested_mcp_props.items() if isinstance(np_details, dict)
                                        }
                                gemini_tool_params_schema["properties"][param_name] = current_param_gemini_schema
                        
                        mcp_required_list = mcp_input_schema.get("required")
                        if isinstance(mcp_required_list, list):
                            gemini_tool_params_schema["required"] = mcp_required_list
                
                if gemini_tool_params_schema and gemini_tool_params_schema.get("type") != "OBJECT":
                        print(f"Info (Tool: {tool_name_mcp}): Tool's input schema type is '{gemini_tool_params_schema.get('type')}', not OBJECT. Gemini FunctionDeclaration.parameters expects an OBJECT schema or None. Setting parameters to None for this tool.")
                        gemini_tool_params_schema = None


                functional_tools.append(
                    types.FunctionDeclaration(
                        name=tool_name_mcp,
                        description=tool_description_mcp,
                        parameters=gemini_tool_params_schema, 
                        # behavior='NON_BLOCKING'
                    )
                )
        else:
            print("No tools found from any MCP server response or 'all_mcp_tools' is empty.")

        # print(functional_tools)

        tools = [
            types.Tool(
                function_declarations=functional_tools,
                # code_execution={},
                google_search={}
            )
        ]
        
        CONFIG = types.LiveConnectConfig(
            response_modalities=[
                "AUDIO",
            ],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                language_code="en-US",
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            tools=tools,
        )
        
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()    
    main_audio_loop = AudioLoop(video_mode=args.mode)
    asyncio.run(main_audio_loop.run())
