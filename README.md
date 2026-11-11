# eventLogSimulation

LLM-based multi-agent simulation for generating synthetic business process event logs.

A BPMN-style process description (see `configs/`) is executed by LLM agents that play the
organisation's employees and external parties. A coordinator agent hands work between them,
and every executed activity is written to an event log (CSV) with case attributes, event
attributes and resources. Case arrivals follow a non-homogeneous Poisson process with
monthly, weekday and hourly rates. The generated logs can then be compared with real logs
using process-mining distance measures.

## Project structure

| Path | Contents |
|------|----------|
| `main.py` | Example run: simulates a P2P (3-way matching) process |
| `simulation_src/` | Simulation engine, agents, coordinator, process manager, prompt and tool management, persistence (TinyDB / SQLite) |
| `configs/` | Process configurations (process summary, roles, case/event attributes) |
| `prompts/` | Prompt templates (YAML) and few-shot examples (JSON) |
| `evaluate.py`, `evaluate_utils.py`, `metrics.py` | Comparing generated logs with real logs (control-flow, n-gram, temporal and workforce distances, trace variants, attribute distributions) |
| `analyse_real_log.py`, `explore_real_log.py` | Exploring real event logs |
| `bpmn_files/`, `images/`, `excalidraw_files/` | Example models and diagrams |

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file in the project root with your API keys (it is git-ignored):

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
```

## Usage

```bash
uv run main.py
```

Outputs are written to `generated_event_logs/`, `protocols/`, `db/` and `arrivals/`.
Adjust the config path, number of traces and arrival rates in `main.py`.

## Data

The evaluation scripts use public event logs from the BPI Challenges (e.g. BPI Challenge
2019), available from the [4TU.ResearchData](https://data.4tu.nl/) repository under
CC BY 4.0. They are not included in this repository; download them into `real_event_logs/`.
Please cite the original datasets when using them.

## License

[MIT](LICENSE)
