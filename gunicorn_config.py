# Gunicorn config file
# --bind: The socket to bind to.
# --workers: The number of worker processes for handling requests.
# --timeout: Workers silent for more than this many seconds are killed and restarted.

bind = "0.0.0.0:5000"
workers = 4
timeout = 120
