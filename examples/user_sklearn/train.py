"""An ordinary sklearn script — NO flashruntime import anywhere.

FlashRuntime's whole contract with this file is convention:
CLI flags in, metrics.json out.

    python train.py --model rf --n_estimators 100        # by hand
    examples/bring_your_code_demo.py                     # operated sweep
"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=50)
    args = parser.parse_args()

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=600, n_features=12, random_state=0)
    if args.model == "logreg":
        estimator = LogisticRegression(C=args.C, max_iter=500, random_state=0)
    elif args.model == "rf":
        estimator = RandomForestClassifier(n_estimators=args.n_estimators, random_state=0)
    else:
        raise SystemExit(f"unknown model {args.model!r} (logreg|rf)")

    scores = cross_val_score(estimator, X, y, cv=3)
    metrics = {
        "model": args.model,
        "C": args.C,
        "n_estimators": args.n_estimators,
        "accuracy_mean": round(float(scores.mean()), 4),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
