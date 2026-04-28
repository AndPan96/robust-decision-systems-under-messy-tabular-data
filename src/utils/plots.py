import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_train_plot(result, save_path):

    training_curves = result["training_curves"]
    validation_curves = result["validation_curves"]

    if not training_curves or not validation_curves: return
    if not all(len(fold) > 0 for fold in training_curves): return

    train_matrix = np.array(training_curves).mean(axis=2)
    train_mean = train_matrix.mean(axis=0)   
    train_std = train_matrix.std(axis=0)
    X_train = np.arange(train_mean.shape[0])

    val_steps = [step for step, _ in validation_curves[0]]
    val_values = []
    for fold in validation_curves:
        val_values.append([loss for _, loss in fold])
    val_mean = np.mean(val_values, axis=0)
    val_std = np.std(val_values, axis=0)

    plt.figure(figsize=(10,6))

    plt.plot(X_train, train_mean, label="Train loss")
    plt.fill_between(X_train, train_mean - train_std, train_mean + train_std, alpha=0.2)

    plt.plot(val_steps, val_mean, label="Val loss", marker="o")
    plt.fill_between(val_steps, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.title(f"Training diagnostics - {result["name"]}")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results, save_path):

    if len(results) <= 1: return

    names = []
    means = []
    stds = []

    prior_mean = None

    for r in results:
        names.append(r["name"])
        means.append(r["metrics"]["accuracy_mean"])
        stds.append(r["metrics"]["accuracy_std"])

        if r["model_class_name"] == "PriorModel":
            prior_mean = r["metrics"]["accuracy_mean"]

    sorted_idx = np.argsort(means)[::-1]
    names = [names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    best_idx = 0

    x = np.arange(len(names))

    plt.figure(figsize=(10, 6))

    for idx in range(len(results)):
        if idx == best_idx:
            plt.errorbar(
                x[idx],
                means[idx],
                yerr=stds[idx],
                fmt='*',
                markersize=14,
                capsize=5,
                label="Models"
            )
        else:
            plt.errorbar(
                x[idx],
                means[idx],
                yerr=stds[idx],
                fmt='o',
                capsize=5
            )

    plt.xticks(x, names, rotation=45, ha="right")

    if prior_mean is not None:
        plt.axhline(
            y=prior_mean,
            linestyle='--',
            label="PriorModel baseline"
        )

    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_monitoring_plot(metrics_path, save_path):

    df = pd.read_parquet(metrics_path)

    if df.empty:
        return

    df = df.sort_values("timestamp")

    x = range(len(df))
    y = df["accuracy"]

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label="Accuracy", marker="o")

    plt.title("Monitoring Accuracy Over Time")
    plt.xlabel("Monitoring steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
        
        