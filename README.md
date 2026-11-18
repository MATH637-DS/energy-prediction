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
## Repository Structutre
  1. **/artifacts:** for saved models once we start training
  2. **/backend:** for any API interaction scripts, **/collection** for pulling **/submission** for result submision
  3. **/dashboard:** frontend components for the live dashboard
  4. **/data:** route data pulled from the api here. going to create subdirectories for weather/power/forecast
  5. **/documentation:** api guide, project description, etc.
  6. **/logs:** route error logs etc.
  7. **/report:** stuff for the final report.
