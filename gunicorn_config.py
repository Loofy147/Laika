# Gunicorn config file

bind = "0.0.0.0:5000"
"""
The socket to bind to.

A string of the form: 'HOST', 'HOST:PORT', 'unix:PATH'.
An IP is a valid HOST.
"""

workers = 4
"""
The number of worker processes for handling requests.

A positive integer generally in the 2-4 x $(NUM_CORES)$ range.
"""

timeout = 120
"""
Workers silent for more than this many seconds are killed and restarted.

Value is a positive number or 0. Setting it to 0 has the effect of
infinite timeouts.
"""
