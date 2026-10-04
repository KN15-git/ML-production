# ML in Production - Iris Classifier API

A FastAPI service that serves a scikit-learn model (Random Forest trained on the
Iris dataset), containerised with Docker and deployed on Render.

**Workflow:** Model -> FastAPI -> Docker -> GitHub -> Render -> Logging & Monitoring

## Live API

- Base URL: `https://YOUR-SERVICE-NAME.onrender.com`
- Interactive docs: `https://YOUR-SERVICE-NAME.onrender.com/docs`

## Endpoints

| Method | Path       | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/`        | Welcome message                               |
| GET    | `/health`  | Health check (also reports if model is loaded)|
| POST   | `/predict` | Predict the Iris species from 4 measurements  |

### Example request

```bash
curl -X POST https://YOUR-SERVICE-NAME.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Example response

```json
{"prediction": 0, "class_name": "setosa"}
```

`features` = `[sepal_length, sepal_width, petal_length, petal_width]` (cm).

## Logging & monitoring

The app logs to stdout, which Render shows in the **Logs** tab:

- model load status at startup
- every request: method, path, status code, latency (ms)
- every prediction: input features and predicted class
- errors (e.g. invalid input -> 422)

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                 # (re)creates model.pkl
uvicorn main:app --reload       # http://127.0.0.1:8000/docs
```

## Run with Docker

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

## Project structure

```
main.py            FastAPI app (endpoints + logging)
train.py           Trains the model and saves model.pkl
model.pkl          Trained Random Forest model
requirements.txt   Pinned dependencies
Dockerfile         Container definition
```
