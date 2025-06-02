# Air Assistant

## Features

-   ⁠Ask anything on your screen
-   ⁠Let AI do the work for you
-   Access to variant helpful mcp tools

## Tech Stacks

-   Gemini
-   OmniParser
-   MCP
-   OpenCV
-   Mediapipe
-   PyQt5

## Run in Local

1. Install `uv` command

    ```bash
    pip install uv
    ```

2. Install dependencies

    ```bash
    uv sync
    ```

3. Create `.env` file by copying `.env.example`

    ```bash
    cp .env.example .env
    ```

4. For running omniparser

    - Setup huggingface CLI
    - Download the weights

        ```bash
        cd omniparser

        # download the model checkpoints to local directory OmniParser/weights/
        for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done

        mv weights/icon_caption weights/icon_caption_florence
        ```

    - In `omniparser/omnitool/client` directory, create `.env` by copying `.env.example`

5. Run the application

    Run omniparserserver:

    ```bash
    uv run omniparser/omnitool/server/omniparserserver.py
    ```

    Run omniparser mcp server:

    ```bash
    uv run omniparser/omnitool/client/mcp_server.py
    ```

    Run main AI client:

    ```bash
    # by default using screen mode
    uv run main.py
    ```

    Run virtual mouse:

    ```bash
    uv run virtual_mouse/virtual_mouse.py
    ```

## Future plan

-   ⁠Increase accuracy
-   ⁠Create friendlier UIs
-   ⁠Add more mcp tools
