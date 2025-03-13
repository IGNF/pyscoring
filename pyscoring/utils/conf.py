import logging
import os
import argparse
import sys
import time
import configparser


class Config:

    def __init__(self):
        """
        Initialisation of the launching of PyScoring by the parsing of the config given (file or args).

        Notes :
            * The config file should (for now) always saved where the script is executed and named 'config.ini'
            * The arguments given by the call to the python file will always overpass the arguments given through the
            config file.
            * The parameters parsed can be divided into 4 categories :
                * MODE : enables to choose either we want to use PyScoring either for 2D or 3D data
                * PATH : defines the different paths where the data is saved and where should the script save its output
                * DATA : some information about the structure and the characteristics of the data
                * PARAMETERS : the definition of the parameters used by PyScoring during this execution


        For more information, you can check the config files given in the example repertories.
        """
        # Time
        self.start = time.time()

        # Read config file
        self.config = "./config.ini"
        MODE, PATH, DATA, PARAM, OUTPUT = self.parse_conf()

        # Parameters
        args = self.parse_args()

        self.mode = args.mode if args.mode else MODE if MODE else 2
        if self.mode == 2:
            self.metrics_families = ["intrinsic", "topo", "edges", "surfacic"]
        else:
            self.metrics_families = ["3d", "topo"]

        # Param paths
        self.gt_path = (
            args.gt_path
            if args.gt_path
            else PATH["GT_PATH"] if PATH["GT_PATH"] else sys.exit("No GT path provided")
        )
        self.p_path = (
            args.pred_path
            if args.pred_path
            else PATH["P_PATH"] if PATH["P_PATH"] else sys.exit("No P path provided")
        )
        self.outpath = (
            args.output
            if args.output
            else PATH["OUTPUT_PATH"] if PATH["OUTPUT_PATH"] else "./"
        )

        # Prepare output file paths
        path, outfile = os.path.split(self.outpath)
        if len(outfile) < 4 or outfile[-4] != ".":
            path = os.path.join(path, outfile)
            outfile = outfile + "_output.csv"
        if not os.path.exists(path):
            os.makedirs(path)
        self.outdir = path
        self.outpath = os.path.join(path, outfile)
        self.outpath_building = os.path.join(path, "building_" + outfile)
        self.outpath_total = os.path.join(path, "global_" + outfile)

        # Param about data
        self.resolution = (
            float(args.resolution)
            if args.resolution
            else float(DATA["RES"]) if DATA["RES"] else 0.5
        )
        self.gt_class = (
            str(args.gt_class)
            if args.gt_class
            else str(DATA["GT_CLASS_VAL"]) if DATA["GT_CLASS_VAL"] else None
        )
        self.p_class = (
            str(args.p_class)
            if args.p_class
            else str(DATA["P_CLASS_VAL"]) if DATA["P_CLASS_VAL"] else None
        )
        self.gt_id = (
            str(args.gt_id)
            if args.gt_id
            else str(DATA["GT_ID_COL"]) if DATA["GT_ID_COL"] else None
        )
        self.p_id = (
            str(args.p_id)
            if args.p_id
            else str(DATA["P_ID_COL"]) if DATA["P_ID_COL"] else None
        )
        self.w_bbox_source = (
            str(args.w_bbox)
            if args.w_bbox
            else str(DATA["WBBOX_PATH"]) if DATA["WBBOX_PATH"] else None
        )
        self.start_index = -9999999
        self.no_data = -9999

        # Options and calculation options
        self.strict_bool = (
            args.strict_bbox
            if args.strict_bbox
            else bool(PARAM["STRICT_BBOX"]) if PARAM["STRICT_BBOX"] else False
        )
        self.epsg = args.epsg if args.epsg else PARAM["EPSG"] if PARAM["EPSG"] else None
        self.extended = (
            args.extend_matches
            if args.extend_matches
            else PARAM["EXTEND_MATCHES"] if PARAM["EXTEND_MATCHES"] else False
        )
        self.nb_workers = (
            int(args.workers)
            if args.workers
            else int(PARAM["WORKERS"]) if PARAM["WORKERS"] else 1
        )
        self.criteria = str(PARAM["CRT"]) if PARAM["CRT"] else "ioma"
        self.threshold = float(PARAM["TSD"]) if PARAM["TSD"] else 0.5

        # TODO 3D

        # Output
        self.log = (
            args.log if args.log else bool(OUTPUT["LOG"]) if OUTPUT["LOG"] else True
        )
        self.local_output = (
            args.localm
            if args.localm
            else bool(OUTPUT["LOCAL"]) if OUTPUT["LOCAL"] else True
        )
        self.global_output = (
            args.globalm
            if args.globalm
            else bool(OUTPUT["GLOBAL"]) if OUTPUT["GLOBAL"] else True
        )
        self.rasters = True

        # Prepare log out
        if self.log:
            self.set_log(self.outpath)

        if not sys.warnoptions:
            import warnings

            warnings.simplefilter("ignore")

    def parse_conf(self):
        """ "
        Parses the arguments given through the config file saved at ./config.ini
        """
        conf_parser = configparser.ConfigParser()
        conf_parser.read(self.config)

        mode = None
        paths_dic = {"GT_PATH": None, "P_PATH": None, "OUTPUT_PATH": None}
        data_dic = {
            "GT_CLASS_COL": None,
            "GT_CLASS_VAL": None,
            "P_CLASS_COL": None,
            "P_CLASS_VAL": None,
            "GT_ID_COL": None,
            "P_ID_COL": None,
            "RES": None,
            "WBBOX_PATH": None,
        }
        para_dic = {
            "STRICT_BBOX": None,
            "EPSG": None,
            "EXTEND_MATCHES": None,
            "WORKERS": None,
            "CRT": None,
            "TSD": None,
        }
        out_dic = {"LOG": None, "LOCAL": None, "GLOBAL": None}

        if "MODE" in conf_parser:
            if "MODE" in conf_parser["MODE"]:
                mode = int(conf_parser["MODE"]["MODE"])

        if "PATHS" in conf_parser:
            for key in list(paths_dic):
                if key in conf_parser["PATHS"]:
                    paths_dic[key] = conf_parser["PATHS"][key]

        if "DATA" in conf_parser:
            for key in list(data_dic):
                if key in conf_parser["DATA"]:
                    data_dic[key] = conf_parser["DATA"][key]

        if "PARAMETERS" in conf_parser:
            for key in list(para_dic):
                if key in conf_parser["PARAMETERS"]:
                    para_dic[key] = conf_parser["PARAMETERS"][key]

        if "OUTPUT" in conf_parser:
            for key in list(out_dic):
                if key in conf_parser["OUTPUT"]:
                    out_dic[key] = conf_parser["OUTPUT"][key]

        return mode, paths_dic, data_dic, para_dic, out_dic

    def parse_args(self):
        """ "
        Parses the arguments given through the call to the python script.
        """
        parser = argparse.ArgumentParser(
            "Comparison of LoD0 shapefile buildings models " "and metrics production"
        )
        parser.add_argument(
            "--gt_path", "-g", help="Path to ground truth shapefile file"
        )
        parser.add_argument(
            "--pred_path", "-p", help="Path to prediction shapefile file"
        )
        parser.add_argument("--output", "-o", help="Path to the output csv")
        parser.add_argument("--resolution", "-r", help="Output raster resolution")
        parser.add_argument("--gt_class", help="Id of the class of interest in the GT")
        parser.add_argument(
            "--p_class", help="Id of the class of interest in the prediction"
        )
        parser.add_argument(
            "--gt_id",
            help="Column of the id of the instances, if None no ID will be provided",
        )
        parser.add_argument(
            "--p_id",
            help="Column of the id of the instances, if None no ID will be provided",
        )
        parser.add_argument(
            "--w_bbox",
            help="Take the first shape of the shapefile in input as working bbox, "
            "if none works with the bbox of input p and gt",
        )
        parser.add_argument(
            "--strict_bbox",
            help="Exclude or not surfaces touching the border of the working bbox",
        )
        parser.add_argument(
            "--epsg",
            help="Projected coordinates system to use. "
            "If not given, will use the epsg of the data, "
            "if not projected will throw an error.",
        )
        parser.add_argument(
            "--extend_matches", help="Extend matches to union of buildings"
        )
        parser.add_argument(
            "--workers",
            help="Number of workers used during the parallelization process."
            " Warning : None = Max",
        )
        parser.add_argument("--log", help="Use pyscoring logger or not")
        parser.add_argument("--localm", help="Enable local metrics output")
        parser.add_argument("--globalm", help="Enable global metrics output")
        parser.add_argument("--mode")
        return parser.parse_args()

    def set_log(self, outpath):
        path, outfile = os.path.split(outpath)

        logging.basicConfig(
            filename=os.path.join(path, "log_" + outfile[:-3] + "log"),
            filemode="w",
            format="%(name)s → %(levelname)s: %(message)s",
        )

        self.log = True

    def pys_print(self, string2print):
        """
        A function to print into the console and into a logging file.

        :param string2print: str, the string to print
        """
        print(string2print)
        if self.log:
            logging.warning(string2print)

    def time(self, start=None, end=None, label="Total"):
        """
        A special print of the time to check how many time does this execution takes
        :param start: start time
        :param end: time to check now
        :param label: the label to print with the result of time check
        :return:
        """
        if not end:
            end = time.time()
        if not start:
            start = self.start
        total_time = round(end - start, 2)
        # str_time = str(total_time) if total_time < 60 else str(round(total_time/60, 2))
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        h = round(h, 2)
        m = round(m, 2)
        s = round(s, 2)
        str_time = (
            str(h) + " h " + str(m) + " m " + str(s) + " s"
            if h
            else str(m) + " m " + str(s) + " s" if m else str(s) + " s"
        )
        self.pys_print(label + " time: " + str_time)

    def metrics(self):
        None
        # TODO function to enable or disable a family of metrics
