# CO2 Forecasting Streamlit App

This project is ready to run and deploy as a Streamlit app using:
- `app_advanced.py` (entrypoint)
- `requirements.txt` (Python dependencies)
- `runtime.txt` (Python runtime for Streamlit Cloud)

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your API key and optional model name for the AI advisor feature:

```bash
cp .env.example .env
```

PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY` and, if desired, `OPENAI_MODEL`.

You can also use Streamlit secrets instead of a local `.env` file by copying `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.

4. Start the app:

```bash
streamlit run app_advanced.py
```

## Deploy To Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud: https://share.streamlit.io
3. Click **New app** and select:
- Repository: your GitHub repo
- Branch: `main` (or your deployment branch)
- Main file path: `app_advanced.py`
4. In app settings, add these secrets:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
```

5. Deploy.

If `OPENAI_API_KEY` is not configured, the app still works and falls back to rule-based suggestions for advisor features.
