import sys
import pyscoring

def main():
    # Load args by parsing
    config = pyscoring.utils.Config()

    # Load ground truth and prediction surfaces and bbox
    dataLoader = pyscoring.utils.DataLoader(config)
    dataLoader.load_data()

    # Tiling the data by a first match to enable the parallelization
    dataLoader.tiling()

    # Metrics
    metrics_launcher = pyscoring.utils.Launcher(config, dataLoader)

    # Pyscoring's metric computers
    metrics_launcher.build_computers()

    # Computation of the metrics by launching the parallelisation on the input entities
    metrics_launcher.launch_parallelization()

    # Merge the metrics and put in output files
    metrics_launcher.print_out()
    config.time()

if __name__ == '__main__':
    sys.exit(main())
