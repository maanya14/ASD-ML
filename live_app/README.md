# ASD-ML Live App

Run from the project root:

```bash
uvicorn live_app.app:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The browser captures live webcam and microphone features, sends lightweight features to the FastAPI backend, and replays physiologic probabilities from `PHYSIO/WESAD_OVERLOAD_RESULTS.csv`.
