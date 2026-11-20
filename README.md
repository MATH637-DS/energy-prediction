# energy-prediction

## Project Description
This project focuses on predicting energy demand of five University of Delaware Buildings

| Building   | API ID     | Type               |
|------------|------------|--------------------|
| Gore Hall  | classroom  | Classroom building |
| Brown Hall | residence  | Residence hall     |
| Building X | laboratory | Research laboratory|
| Lil Bob Gym| recreation | Recreation center |
| Morris Library| library | Library |

To do so, we rely on historical data from the provided API:
<https://math637-energy-api.dominique637.workers.dev/>

We use a collection of different learning methods to make our predicitons

## Project Setup
This project uses Python >=3.12
1. **Clone the repository**
```bash
git clone https://github.com/jhcooper/energy-prediction.git

```
2. **Install uv**

- macOS: `brew install uv`
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Any OS (fallback): `pip install uv`

3. **Sync Project**
```bash
cd energy-prediction 
uv sync
```

### Notes on UV

UV is a useful all-in-one tool that's used to manage python projects. It provides package installation, dependency version control, and virtual environment managemnet in a single package thats very quick and easy to use. 

#### 1. Adding Dependencies
To install a new library(s) in the project, you use `uv add`, i.e.
 ```bash
 uv add pandas numpy scikit-learn
```
uv will automatically resolve version conflicts and add all dependencies to [uv.lock](uv.lock) to make sure that all team members have exactly the same versions.
#### 2. Syncing After A Pull
uv does this automatically when using `uv run`, but it doesnt hurt to do at the beginning of a session since it only takes seconds. running `uv sync` after pulling, i.e.
```bash
git pull origin main
uv sync
```
makes sure that any dependencies added by someone else are downloaded in your environment

#### 3. Running A Script
Pretty Simple, just use
```bash
uv run path-to-file.py
```

## Connecting to the API
To connect to the API you need to create a .env file and add the API key from email. You can do so by copying .env.example
```bash
cp .env.example .env
nano .env
```
or just create the file yourself, and add/replace
```text
API_KEY=YOUR_API_KEY
```

## Forecast dashboard
The static website in [`docs/`](docs/) is deployed to GitHub Pages by the
workflow in [`.github/workflows/pages.yml`](.github/workflows/pages.yml). In
the repository settings, set **Pages → Build and deployment → Source** to
**GitHub Actions**. The first deployment runs after a push to `main` that
changes the website, trained artifacts, observations, or dashboard generator;
it can also be started manually from the Actions tab.

The dashboard reads its public, static data bundle from
[`docs/data/forecasts.json`](docs/data/forecasts.json). Timestamps must be ISO
8601, and all demand and accuracy values are in kilowatts. `actual_kw` may be
`null` until target readings arrive. Keep API keys out of this file; it is
published publicly with the site.

```json
{
  "updated_at": "2026-11-20T11:55:00Z",
  "historical_demand": [
    {
      "timestamp": "2026-11-20T17:45:00Z",
      "building_id": "classroom",
      "demand_kw": 180.1
    }
  ],
  "models": [
    {
      "id": "linear_regression_6h",
      "name": "Linear regression · 6-hour",
      "horizon_hours": 6,
      "lookback_hours": 24,
      "generated_at": "2026-11-20T18:39:00Z",
      "origin_time": "2026-11-20T17:00:00Z",
      "input_latest_observation": "2026-11-20T16:45:00Z",
      "imputed_history_points": 1,
      "training_evaluation": {
        "test_mae_kw": 19.15,
        "test_rmse_kw": 24.87,
        "test_r2": 0.464,
        "per_building": [
          {
            "building_id": "classroom",
            "mae_kw": 26.52,
            "rmse_kw": 33.4,
            "r2": 0.744
          }
        ]
      },
      "forecasts": [
        {
          "target_time": "2026-11-20T23:00:00Z",
          "building_id": "classroom",
          "horizon_hours": 6,
          "prediction_kw": 182.3
        }
      ]
    }
  ],
  "accepted_forecasts": [
    {
      "target_time": "2026-11-20T18:00:00Z",
      "origin_time": "2026-11-20T12:00:00Z",
      "building_id": "classroom",
      "horizon_hours": 6,
      "model_version": "linear-regression-v1",
      "accepted_at": "2026-11-20T11:50:00Z",
      "prediction_kw": 181.2,
      "actual_kw": null
    }
  ],
  "performance": {
    "evaluation_period": {
      "start": "2026-11-13T00:00:00Z",
      "end": "2026-11-20T00:00:00Z"
    },
    "overall": {
      "mae_kw": 8.4,
      "rmse_kw": 11.2,
      "scored_predictions": 56,
      "submitted_predictions": 60,
      "eligible_predictions": 65
    },
    "metrics": [
      {
        "building_id": "classroom",
        "horizon_hours": 6,
        "mae_kw": 7.1,
        "rmse_kw": 9.3,
        "scored_predictions": 12
      }
    ]
  },
  "system_status": {
    "latest_collected_reading": {
      "timestamp": "2026-11-20T11:45:00Z",
      "demand_kw": {
        "classroom": 180.1,
        "residence": 132.4,
        "laboratory": 267.3,
        "recreation": 176.8,
        "library": 164.5
      }
    },
    "latest_accepted_submission": {
      "accepted_at": "2026-11-20T11:50:00Z",
      "model_version": "linear-regression-v1"
    },
    "next_submission_deadline": "2026-11-20T12:00:00Z",
    "errors": []
  }
}
```

The Pages workflow regenerates the bundle from
[`data/observations.jsonl`](data/observations.jsonl) and the four trained
artifacts in [`artifacts/baseline/`](artifacts/baseline/) whenever those inputs
or dashboard code change. To generate it locally, run:

```bash
uv run --frozen python -m energy_prediction.dashboard
```

The dashboard shows historical demand, model forecasts, API-accepted forecasts
and their actuals, recent accuracy and submission coverage, and system status.
Building and historical-range controls filter the views, while the forecast
control covers 1 hour to 1 week. The current trained models produce 6-hour and
24-hour forecasts only; the line connects those available predictions and does
not imply predictions at the hours between them. The inference script may
interpolate one missing 15-minute input reading.

The page reloads the static bundle every five minutes, but it does not collect
new API data while open. A new deployment is needed for updated observations
and predictions. API-accepted submissions, their matched actuals, and recent
scoreboard metrics remain empty until those API results are saved and added to
the bundle; generated model forecasts are never labeled as API-accepted.

## Repository Structutre
  1. **/artifacts:** for saved models once we start training
  2. **/backend:** for any API interaction scripts, **/collection** for pulling **/submission** for result submision
  3. **/docs:** static forecast dashboard and GitHub Pages data
  4. **/data:** route data pulled from the api here. going to create subdirectories for weather/power/forecast
  5. **/documentation:** api guide, project description, etc.
  6. **/logs:** route error logs etc.
  7. **/report:** stuff for the final report.
