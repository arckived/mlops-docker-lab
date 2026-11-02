# Wine Quality Sommelier (Docker Lab)

A model trained on the **UCI Wine Quality** dataset rates a wine as **Low, Medium or High quality**
from its lab measurements. The app runs as **three containers** with Docker Compose: a trainer, a web app,
and a Postgres database that logs every prediction.

![Prediction dashboard](docs/dashboard.png)


## How it works

```
                docker compose up
                        │
   ┌────────────────────┼──────────────────────┐
   ▼                    ▼                      │
 db (postgres)     trainer (one-shot)          │
 healthcheck       trains 2 models, saves      │
 pg_isready        best → model_artifacts vol  │
   │                    │ exits 0              │
   └──── healthy ───────┴── completed ─────────┘
                        ▼
                  app (gunicorn + Flask)
                  reads model_artifacts (read-only)
                  writes predictions → db
                  http://localhost:5000
```

* **`trainer`** is built from the `trainer` stage of the Dockerfile. It trains the models, writes
  `model.joblib` and `metrics.json` into the `model_artifacts` named volume, and exits.
* **`db`** is `postgres:16-alpine` with a healthcheck, storing data in the `pg_data` volume.
* **`app`** waits until the trainer has **finished successfully** and the database is **healthy**
  (`depends_on` conditions), then serves the model from the shared volume.

The web pages are plain HTML forms rendered by Flask (Jinja templates). There is no custom CSS or
JavaScript: the form posts to Flask, Python runs the model, and Flask returns the page with the result.

The Dockerfile has four stages:

1. `base`: slim Python image and dependencies (shared by the other stages, so they're installed once)
2. `trainer`: adds the data and training script (used by compose)
3. `model_build`: runs training during `docker build`
4. `serving`: the web app, with the model from `model_build` copied in

Because of stage 3, the serving image works on its own with plain `docker run` (it then logs to SQLite inside
the container). With compose, the volume mount replaces the built-in model with the one the trainer container
just produced.

## Project layout

```
.
├── Dockerfile            multi-stage: base → trainer → model_build → serving
├── docker-compose.yml    db + trainer + app
├── requirements.txt
├── data/                 UCI Wine Quality CSVs (red and white)
├── src/
│   ├── train.py          training, model selection, metrics.json
│   ├── app.py            Flask routes and API
│   ├── db.py             prediction log (Postgres, or SQLite fallback)
│   └── templates/        plain HTML pages: predict, model, history
└── docs/                 screenshots
```

## Run it

### Option A: Docker Compose (all three containers)

```bash
docker compose up --build
```

Open <http://localhost:5000>. The first start takes about 20 seconds while the trainer runs.

To use a different port:

```bash
APP_PORT=8000 docker compose up --build
```

Stop everything and delete the volumes (model and logged predictions):

```bash
docker compose down -v
```

### Option B: Single image with `docker build` and `docker run`

```bash
docker build -t wine-sommelier .
```

```bash
docker run -p 5000:5000 wine-sommelier
```

### Option C: Without Docker

```bash
pip install -r requirements.txt
python src/train.py
cd src && python app.py
```

## API

Predict:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"wine_type": "red", "fixed acidity": 7.9, "volatile acidity": 0.35, "citric acid": 0.46,
       "residual sugar": 3.6, "chlorides": 0.078, "free sulfur dioxide": 15,
       "total sulfur dioxide": 37, "density": 0.9973, "pH": 3.35, "sulphates": 0.86, "alcohol": 12.8}'
```

```json
{
  "id": 1,
  "wine_type": "red",
  "predicted_class": "High",
  "confidence": 0.5633,
  "probabilities": {"Low": 0.0567, "Medium": 0.38, "High": 0.5633}
}
```

Missing or invalid fields return HTTP 400 with an `error` message.

| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET, POST | Prediction form (POST submits the form and shows the result) |
| `/metrics` | GET | Model evaluation page |
| `/history` | GET | Last 50 logged predictions |
| `/api/predict` | POST | Prediction as JSON (also saved to the database) |
| `/api/metrics` | GET | Contents of `metrics.json` |
| `/api/history?limit=N` | GET | Logged predictions as JSON |
| `/health` | GET | Status, model name, training time (used by the Docker healthcheck) |

Check what is in the database:

```bash
docker compose exec db psql -U sommelier -d predictions -c "select * from predictions order by id desc limit 5"
```

## Results

| Model | Test accuracy | Macro F1 |
|---|---|---|
| **RandomForest** (served) | 0.729 | 0.722 |
| GradientBoosting | 0.681 | 0.669 |

Alcohol is the strongest signal by far, followed by volatile acidity and density.
Most errors are between neighbouring classes (Low vs Medium, Medium vs High), and Low is almost never
confused with High.

![Model page](docs/metrics.png)

![History page](docs/history.png)

## Dataset

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from
physicochemical properties.* Decision Support Systems, 47(4):547 to 553, 2009.
Available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).
