from src.experiments import run_all_experiments
from src.visualization import plot_metric_charts, plot_time_charts


def main():
    run_all_experiments()
    plot_metric_charts()
    plot_time_charts()


if __name__ == "__main__":
    main()