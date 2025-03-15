To run:
0) Install Python https://www.python.org/
1) Install Poetry, https://python-poetry.org/
2) Run poetry install in the /api and /web folders.
3) Run /api/run.sh, then /web/run.sh

Docker files are out of date!

For deployment, one must setup a reverse proxy to route requests from the browser client of /_temp_localhost_data to the API url (http://localhost:8000), also uncomment #disableWidgetStateDuplicationWarning = true in /web/.streamlit/config.toml

.env in web folder:
BACKEND_URL=http://localhost:8000

.env in api folder
OPENAI_API_KEY=""
HOST=localhost
PORT=8000
\# DIVIDE ORIGINS BY COMMA
ALLOWED_ORIGINS=localhost:8501