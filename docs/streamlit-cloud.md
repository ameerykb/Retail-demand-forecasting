# Hosting the demo on Streamlit Community Cloud

The demo (`app/streamlit_app.py`) only calls the API, so it is light and free to host.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repository `ameerykb/Retail-demand-forecasting`, branch `main`, main file path `app/streamlit_app.py`.
4. Optional: choose a custom subdomain, e.g. `rossmann-forecast`.
5. **Deploy**. Streamlit installs `app/requirements.txt` (next to the app), not the heavy root requirements.

The app calls the live AWS API by default. To point it elsewhere, add a secret in the app settings:

```toml
API_URL = "https://<your-api>.execute-api.<region>.amazonaws.com"
```

Apps on the free tier sleep after a few days without visitors; opening the link wakes them up.
