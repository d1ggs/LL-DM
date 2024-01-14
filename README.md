# LL-DM

This project aims at creating a tool to generate and host a campaign for the game Dungeons and Dragons 5th edition.

## TODO
* [x] Create a basic UI
* [ ] Allow file upload
* [ ] Set up document retrieval
* [ ] Set up chat summarization
* [ ] Set up audio generation (possible memory constraints)

## Installation
The following instructions are for AMD cards. I will provide an NVIDIA version soon.

0. This project uses PDM as a package manager
```bash
pip install pipx
pipx install pdm
```

1. Install the dependencies
```bash
pdm install
```

3. Start the UI
```bash
pdm run chainlit run app.py
```