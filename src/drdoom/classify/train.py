"""Train and score the root-cause classifier.

Run with ``python -m drdoom.classify.train``.

Three decisions shape this module.

**The classifier has its own split.** The detection pipeline deliberately trains on an
anomaly-free period, which contains no examples of anything to classify. Classification
is therefore split over the labelled data only, holding out whole machines or services so
that a test incident comes from a system the model has never seen.

**Folds are grouped by event.** Windows are cut every ten minutes from incidents lasting
hours, so a random fold split puts overlapping slices of one outage on both sides of the
boundary and reports memorisation as skill. ``GroupKFold`` over event id prevents it.

**Scores are reported per event.** Window predictions are pooled by majority vote into a
single verdict per incident, which is the unit an engineer acts on.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from drdoom.classify import archetypes as archetype_module
from drdoom.classify import dataset as dataset_module
from drdoom.classify.card import write_card
from drdoom.classify.features import FeatureSpec
from drdoom.config import get_settings
from drdoom.data import smd, synthetic
from drdoom.data.schema import MetricSeries
from drdoom.data.splits import SplitResult, held_out_series_split
from drdoom.data.windows import build_index

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SOURCES = ("smd", "synthetic")


@dataclass(frozen=True)
class ClassifierConfig:
    source: str = "synthetic"
    window_size: int = 60
    stride: int = 10
    n_trials: int = 30
    n_folds: int = 5
    seed: int = 42
    n_scenarios: int = 200
    days: int = 4
    split_seed_search: int = 40
    models_root: Path | None = None
    docs_root: Path | None = None

    @property
    def model_dir(self) -> Path:
        root = self.models_root or get_settings().models_dir
        return root / "classifier" / self.source


@dataclass
class Prepared:
    splits: SplitResult
    label_names: tuple[str, ...]
    archetypes: archetype_module.Archetypes | None
    notes: dict = field(default_factory=dict)


def labelled_series(config: ClassifierConfig) -> list[MetricSeries]:
    """Return the series that actually carry labelled incidents."""
    if config.source == "smd":
        return [labelled for _, labelled in smd.load_all().values()]
    return synthetic.generate(n_scenarios=config.n_scenarios, days=config.days, seed=config.seed)


def _class_coverage(splits: SplitResult, lookup: dict[int, str], names: tuple[str, ...]) -> int:
    """Smallest number of events any class has in any split; zero means a blind split."""
    worst = np.inf
    for items in splits.as_dict().values():
        counts = Counter(
            lookup.get(event.event_id, "other") for item in items for event in item.events
        )
        for name in names:
            worst = min(worst, counts.get(name, 0))
    return int(worst)


def choose_split(
    series: list[MetricSeries], lookup: dict[int, str], names: tuple[str, ...], attempts: int
) -> tuple[SplitResult, int]:
    """Pick the series partition that leaves every class present in every split.

    Holding out whole machines can easily strand a class entirely on one side. Rather
    than accept a split that makes a class unscorable, several partitions are tried and
    the one with the best worst-case class coverage is kept.
    """
    best_split, best_seed, best_score = None, 0, -1
    for seed in range(attempts):
        candidate = held_out_series_split(series, 0.6, 0.15, seed=seed)
        score = _class_coverage(candidate, lookup, names)
        if score > best_score:
            best_split, best_seed, best_score = candidate, seed, score
        if score >= 5:
            break
    assert best_split is not None
    return best_split, best_seed


def prepare(config: ClassifierConfig) -> Prepared:
    """Load the labelled data, decide the label set, and split it."""
    series = labelled_series(config)
    events = [event for item in series for event in item.events]

    if config.source == "smd":
        # Archetypes are a taxonomy derived from the annotation file, not from metric
        # values, so defining them over all events does not hand the model any signal it
        # could read off its inputs. With 325 incidents in total there is not enough data
        # to define the taxonomy on a subset and still have it be stable.
        derived = archetype_module.derive(events, smd.N_FEATURES)
        names = tuple(name for name in derived.names if name != archetype_module.OTHER_LABEL)
        lookup = dataset_module.event_labels(series, derived)
        notes = {"archetype_support": derived.support, "archetype_profiles": derived.profiles}
    else:
        derived = None
        names = tuple(sorted(synthetic.ROOT_CAUSES))
        lookup = dataset_module.event_labels(series, None)
        notes = {}

    splits, seed = choose_split(series, lookup, names, config.split_seed_search)
    notes["split_seed"] = seed
    notes["class_coverage"] = _class_coverage(splits, lookup, names)
    return Prepared(splits=splits, label_names=names, archetypes=derived, notes=notes)


def build_matrices(
    config: ClassifierConfig, prepared: Prepared
) -> dict[str, dataset_module.ClassificationData]:
    matrices = {}
    for name, series in prepared.splits.as_dict().items():
        index = build_index(series, config.window_size, config.stride)
        matrices[name] = dataset_module.build(
            series, index, prepared.label_names, prepared.archetypes
        )
    return matrices


def _objective(
    trial: optuna.Trial,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_folds: int,
    seed: int,
) -> float:
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 80, 400),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    scores = []
    for train_rows, valid_rows in GroupKFold(n_splits=n_folds).split(features, labels, groups):
        model = _make_model(params, n_classes, seed)
        model.fit(
            features[train_rows],
            labels[train_rows],
            sample_weight=compute_sample_weight("balanced", labels[train_rows]),
            verbose=False,
        )
        scores.append(
            f1_score(labels[valid_rows], model.predict(features[valid_rows]), average="macro")
        )
    return float(np.mean(scores))


def _make_model(params: dict, n_classes: int, seed: int) -> xgb.XGBClassifier:
    """Build the estimator, letting xgboost pick binary or multiclass for itself.

    Forcing multi:softprob for a two-class problem makes predict return a probability
    matrix rather than labels, which the metrics then reject.
    """
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    metric = "mlogloss" if n_classes > 2 else "logloss"
    return xgb.XGBClassifier(
        **params,
        objective=objective,
        eval_metric=metric,
        random_state=seed,
        n_jobs=-1,
    )


def event_predictions(
    labels: np.ndarray, predicted: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pool window predictions into one verdict per incident by majority vote."""
    truth, verdict = [], []
    for event_id in np.unique(groups):
        rows = groups == event_id
        truth.append(int(Counter(labels[rows].tolist()).most_common(1)[0][0]))
        verdict.append(int(Counter(predicted[rows].tolist()).most_common(1)[0][0]))
    return np.array(truth), np.array(verdict)


def _bootstrap_macro_f1(
    truth: np.ndarray, verdict: np.ndarray, n_samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    if not len(truth):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(truth), size=(n_samples, len(truth)))
    values = [f1_score(truth[row], verdict[row], average="macro", zero_division=0) for row in draws]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def train(config: ClassifierConfig) -> dict:
    """Tune, fit and score one classifier, and write its card."""
    prepared = prepare(config)
    matrices = build_matrices(config, prepared)
    names = prepared.label_names
    logger.info("%s classes: %s", config.source, list(names))
    for split, data in matrices.items():
        logger.info("  %-5s %5d windows across %3d events", split, len(data), data.n_events)

    tuning_features = np.concatenate([matrices["train"].features, matrices["val"].features])
    tuning_labels = np.concatenate([matrices["train"].labels, matrices["val"].labels])
    tuning_groups = np.concatenate([matrices["train"].groups, matrices["val"].groups])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(
        lambda trial: _objective(
            trial,
            tuning_features,
            tuning_labels,
            tuning_groups,
            len(names),
            config.n_folds,
            config.seed,
        ),
        n_trials=config.n_trials,
    )
    logger.info("best grouped cv macro f1 %.4f", study.best_value)

    model = _make_model(study.best_params, len(names), config.seed)
    model.fit(
        tuning_features,
        tuning_labels,
        sample_weight=compute_sample_weight("balanced", tuning_labels),
        verbose=False,
    )

    test = matrices["test"]
    window_predictions = model.predict(test.features)
    truth, verdict = event_predictions(test.labels, window_predictions, test.groups)

    spec = FeatureSpec(metric_names=test.spec.metric_names)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(config.model_dir / "model.json")
    spec.save(config.model_dir / "features.json")
    (config.model_dir / "labels.json").write_text(json.dumps(list(names)), encoding="utf-8")

    summary = {
        "source": config.source,
        "classes": list(names),
        "best_cv_macro_f1": round(study.best_value, 4),
        "best_params": study.best_params,
        "split_seed": prepared.notes.get("split_seed"),
        "min_events_per_class_per_split": prepared.notes.get("class_coverage"),
        "test_events": len(truth),
        "test_windows": len(test),
        "event_macro_f1": round(float(f1_score(truth, verdict, average="macro")), 4),
        "event_macro_f1_ci": [round(v, 4) for v in _bootstrap_macro_f1(truth, verdict)],
        "event_accuracy": round(float((truth == verdict).mean()), 4),
        "window_macro_f1": round(
            float(f1_score(test.labels, window_predictions, average="macro")), 4
        ),
        "events_per_class": test.events_per_class(),
        "confusion": confusion_matrix(truth, verdict, labels=range(len(names))).tolist(),
        "feature_importance": dict(
            sorted(
                zip(spec.columns, model.feature_importances_.tolist(), strict=True),
                key=lambda item: -item[1],
            )[:12]
        ),
    }
    summary.update({k: v for k, v in prepared.notes.items() if k.startswith("archetype")})
    (config.model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "%s event macro f1 %.3f over %d incidents",
        config.source,
        summary["event_macro_f1"],
        summary["test_events"],
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the root cause classifier.")
    parser.add_argument("--source", choices=[*SOURCES, "both"], default="both")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sources = SOURCES if args.source == "both" else (args.source,)

    summaries = [
        train(
            ClassifierConfig(
                source=source, n_trials=args.trials, n_folds=args.folds, seed=args.seed
            )
        )
        for source in sources
    ]
    write_card(summaries)


if __name__ == "__main__":
    main()
