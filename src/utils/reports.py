from datetime import datetime
from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports")
REPORT_MONITORING_DIR = Path("reports_mntr")

def create_new_report_folder(base_path: Path):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = base_path / timestamp
    plots_path = report_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)
    return report_path, plots_path

def generate_train_report(results, report_path: Path):
    
    html = []
    html.append("<html><head><title>Training Report</title></head><body>")

    html.append("<h1>Model Training Report</h1>")
    html.append(f"<p>Generated at: {datetime.now()}</p>")

    if len(results) > 1:
        comparison_path = report_path / "plots/comparison.png"
        if comparison_path.exists():
            html.append("<h2>Model Comparison</h2>")
            html.append("<img src='plots/comparison.png' width='700'>")

    html.append("<h2>Model Metrics</h2>")
    html.append("<table border='1'>")
    html.append(
        "<tr><th>Model</th><th>Accuracy Mean</th><th>Accuracy Std</th></tr>"
    )

    for r in results:
        html.append(
            f"<tr>"
            f"<td>{r['name']}</td>"
            f"<td>{r['metrics']['accuracy_mean']:.4f}</td>"
            f"<td>{r['metrics']['accuracy_std']:.4f}</td>"
            f"</tr>"
        )

    html.append("</table>")

    for r in results:

        plot_file = f"plots/{r['name']}.png"
        plot_path = report_path / plot_file

        if plot_path.exists():
            html.append(f"<h3>{r['name']}</h3>")
            html.append(f"<img src='{plot_file}' width='600'>")

    html.append("</body></html>")

    with open(report_path / "report.html", "w") as f:
        f.write("\n".join(html))


def generate_monitoring_report(report_path: Path):

    metrics_path = report_path / "metrics.parquet"
    plot_path = report_path / "plots" / "monitoring.png"

    if not metrics_path.exists():
        return

    df = pd.read_parquet(metrics_path)

    if df.empty:
        return

    last_acc = df["accuracy"].iloc[-1]
    mean_acc = df["accuracy"].mean()
    min_acc = df["accuracy"].min()

    html = []
    html.append("<html><head><title>Monitoring Report</title></head><body>")

    html.append("<h1>Model Monitoring Report</h1>")
    html.append(f"<p>Generated at: {datetime.now()}</p>")

    html.append("<h2>Summary</h2>")
    html.append(f"<p>Last accuracy: {last_acc:.4f}</p>")
    html.append(f"<p>Mean accuracy: {mean_acc:.4f}</p>")
    html.append(f"<p>Min accuracy: {min_acc:.4f}</p>")
    html.append(f"<p>Total evaluations: {len(df)}</p>")

    if plot_path.exists():
        html.append("<h2>Accuracy Over Time</h2>")
        html.append("<img src='plots/monitoring.png' width='700'>")

    html.append("</body></html>")

    with open(report_path / "report.html", "w") as f:
        f.write("\n".join(html))

