# remec

Regularized Extended-MHD Equilibrium Code.

The project architecture and staged implementation plan are in
[`docs/DESIGN.md`](docs/DESIGN.md), beginning with its overview in §1.  Contributors
should also read `AGENTS.md` and `WORKFLOW.md` before beginning a milestone.

To interact with the code locally (on my Macbook) use `.venv/bin/python` and
`.venv/bin/pytest`.

Create the environment and install the project with pip:

```bash
python3 -m venv .venv  # Python 3.10 or newer
.venv/bin/python -m pip install -e ".[dev]"
```
