
"""Lance 22 runs d'entraînement dans l'expérience MLflow « scoring »."""

import os
import subprocess
import sys
from itertools import product


CONFIGS = [
    # Référence naïve : ROC AUC attendu autour de 0,5
    ("dummy", []),

    # Régression logistique : 4 x 2 = 8 runs
    *[
        ("logistic_regression", ["--C", str(c), "--class_weight", w])
        for c, w in product(
            [0.01, 0.1, 1.0, 10.0],
            ["none", "balanced"],
        )
    ],

    # Random Forest : 3 x 3 = 9 runs
    *[
        ("random_forest", ["--n_estimators", str(n), "--max_depth", str(d)])
        for n, d in product(
            [100, 200, 400],
            [4, 8, "none"],
        )
    ],

    # Gradient Boosting : 2 x 2 = 4 runs
    *[
        (
            "gradient_boosting",
            ["--n_estimators", str(n), "--max_depth", str(d)],
        )
        for n, d in product(
            [100, 200],
            [2, 3],
        )
    ],
]


def main() -> None:
    """Lance successivement tous les entraînements."""
    print(f"Lancement de {len(CONFIGS)} runs...", flush=True)

    env = {
        **os.environ,
        "LAUNCHED_BY": "grid_search",
    }

    for i, (model, args) in enumerate(CONFIGS, 1):
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--model",
            model,
            *args,
        ]

        print(
            f"[{i}/{len(CONFIGS)}] {' '.join(cmd[1:])}",
            flush=True,
        )

        subprocess.run(cmd, check=True, env=env)

    print("Tous les runs sont terminés.", flush=True)


if __name__ == "__main__":
    main()

