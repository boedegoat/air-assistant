# Air Assistant

Air Assistant is a versatile application designed to enhance your productivity by allowing you to ask and interact with your screen content using AI and a suite of powerful tools.

[Demo Video](https://binusianorg-my.sharepoint.com/personal/bhremada_ardhi_binus_ac_id/_layouts/15/guestaccess.aspx?share=EiQ6CYwk6cNKh9NCX24LDe8Bl9rXvV30OYZSG0q9Z1wLlw)

## Features

-   **Screen Interaction:** Ask questions about anything displayed on your screen.
-   **AI Automation:** Leverage AI to automate tasks and streamline your workflow.
-   **MCP Tools:** Access a variety of Model Context Protocols (MCP) tools for extended functionality.

## Tech Stack

-   Gemini
-   OmniParser
-   MCP
-   OpenCV
-   Mediapipe
-   PyQt5

## Prerequisites

-   Python 3.12
-   `uv`
-   Hugging Face account and CLI setup

## Local Setup and Execution

Follow these steps to get Air Assistant running on your local machine:

1.  **Install `uv` (if not already installed):**

    ```bash
    pip install uv
    ```

2.  **Clone the Repository (if you haven't already):**

    ```bash
    git clone https://github.com/boedegoat/air-assistant.git
    cd air-assistant
    ```

3.  **Install Dependencies:**

    ```bash
    uv sync
    ```

4.  **Configure Environment Variables:**

    -   Create a `.env` file in the project root by copying `.env.example`:
        ```bash
        cp .env.example .env
        ```
    -   Update the `.env` file with your specific configurations (e.g., API keys).

5.  **Set up OmniParser:**

    -   Ensure your Hugging Face CLI is configured (`huggingface-cli login`).
    -   Download OmniParser model weights:
        ```bash
        cd omniparser
        # Download model checkpoints to OmniParser/weights/
        for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
        mv weights/icon_caption weights/icon_caption_florence
        cd ..
        ```
    -   In the `omniparser/omnitool/client` directory, create a specific `.env` file for the OmniParser client:
        ```bash
        cp omniparser/omnitool/client/.env.example omniparser/omnitool/client/.env
        ```
        (Adjust values in `omniparser/omnitool/client/.env`)

6.  **Configure MCP Tools:**

    -   Create `servers_config.json` from `servers_config.example.json` in the project root:
        ```bash
        cp servers_config.example.json servers_config.json
        ```
    -   For the `filesystems` server within `servers_config.json`, adjust the path to your local filesystem as needed.

7.  **Run the Application Components (in separate terminals):**

    -   **OmniParser Server:**

        ```bash
        uv run omniparser/omnitool/server/omniparserserver.py
        ```

    -   **Main AI Client (defaults to screen mode):**

        ```bash
        uv run main.py
        ```

    -   **Virtual Mouse (optional):**
        ```bash
        uv run virtual_mouse/virtual_mouse.py
        ```

## Future Plans

Key areas for future development include:

-   **Accuracy Enhancement:** Continuously improving the precision and reliability of AI features.
-   **User Interface (UI) Improvements:** Developing a more intuitive and user-friendly graphical interface.
-   **Expanded MCP Toolset:** Integrating additional MCP tools to broaden the application's capabilities.
-   **Containerization:** Packaging the application (e.g., using Docker) for easier deployment and scalability.
