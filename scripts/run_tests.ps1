$ErrorActionPreference = "Stop"

# Avoid third-party pytest plugin auto-loading, which may block collection
# or make Ctrl+C unresponsive in some local environments.
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:PYTHONPATH = "src"

python -m pytest tests @args
