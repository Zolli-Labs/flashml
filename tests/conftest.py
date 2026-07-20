"""Test-wide environment setup.

The service module auto-initializes a real app at import time for container
deployments (`uvicorn flashruntime.service.app:app`). Tests construct their
own apps with tmp-path settings, so disable autoinit before anything imports
the module.
"""

import os

os.environ.setdefault("FLASHML_SERVICE_AUTOINIT", "0")
