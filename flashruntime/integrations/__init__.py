"""Framework adapters: each builds CommandWorkloads from one framework's
LAUNCH AND CHECKPOINT CONVENTIONS — never its model code, never a
module-level framework import (four-axes rule). Import the submodule you
need: `from flashruntime.integrations import sklearn, pytorch, huggingface`.
"""
