# Air Assistant

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

    - Put the weights in `omniparser/weights` directory
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
