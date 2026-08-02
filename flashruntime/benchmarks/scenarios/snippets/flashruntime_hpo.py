# flashruntime HPO — minimal setup.
# Source: flashruntime README + examples/bring_your_code_demo.py
import flashruntime as flash
from flashruntime.integrations import sklearn as fr_sklearn

run = flash.submit(
    fr_sklearn.hpo(
        "train.py",
        {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50, 100]},
        source="user_sklearn",
    )
)
print(run.best_trial())
