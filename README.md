# Wine Quality Sommelier (Docker Lab)

A model trained on the **UCI Wine Quality** dataset rates a wine as **Low, Medium or High quality**
from its lab measurements. The app runs as **three containers** with Docker Compose: a trainer, a **Streamlit** dashboard,
and a Postgres database that logs every prediction.

![Predict tab](docs/dashboard.png)


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
                  app (Streamlit dashboard)
                  reads model_artifacts (read-only)
                  writes predictions → db
                  http://localhost:8501
```

* **`trainer`** is built from the `trainer` stage of the Dockerfile. It trains the models, writes
  `model.joblib` and `metrics.json` into the `model_artifacts` named volume, and exits.
* **`db`** is `postgres:16-alpine` with a healthcheck, storing data in the `pg_data` volume.
* **`app`** waits until the trainer has **finished successfully** and the database is **healthy**
  (`depends_on` conditions), then serves the model from the shared volume.

The dashboard is a single Python file, `src/streamlit_app.py`, built with Streamlit. It has three tabs:

* **Predict**: sliders for the 11 measurements and a red/white choice. The sliders start at the typical
  value for the chosen wine type. The result shows the predicted class and the probability of each class.
* **Model**: accuracy, macro F1, scores for each class and the confusion matrix from training.
* **History**: every prediction saved in Postgres.

The Dockerfile has four stages:

1. `base`: slim Python image and dependencies (shared by the other stages, so they're installed once)
2. `trainer`: adds the data and training script (used by compose)
3. `model_build`: runs training during `docker build`
4. `serving`: the Streamlit dashboard, with the model from `model_build` copied in

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
│   ├── streamlit_app.py  Streamlit dashboard (Predict, Model, History tabs)
│   └── db.py             prediction log (Postgres, or SQLite fallback)
└── docs/                 screenshots
```

## Run it

### Option A: Docker Compose (all three containers)

```bash
docker compose up --build
```

Open <http://localhost:8501>. The first start takes about 20 seconds while the trainer runs.

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
docker run -p 8501:8501 wine-sommelier
```

### Option C: Without Docker

```bash
pip install -r requirements.txt
python src/train.py
cd src && streamlit run streamlit_app.py
```

## Check the database

See the saved predictions directly in Postgres:

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

![Model tab](docs/metrics.png)

![History tab](docs/history.png)

## Dataset

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from
physicochemical properties.* Decision Support Systems, 47(4):547 to 553, 2009.
Available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).
