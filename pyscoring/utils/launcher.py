import csv
import time
import fiona
import shapely
from shapely import GEOSException, wkt
from shapely.geometry import mapping
from shapely.ops import transform
from tqdm import tqdm
import concurrent.futures
import rasterio.features

import pyscoring


class Launcher:

    def __init__(self, pyscoring_config, dataloader):

        self.config = pyscoring_config
        self.dataLoader = dataloader
        self.computers = []

        # Output
        self.output = []
        self.ouput_gsurfacic = []

        # Metrics
        self.nbr_gt_matched, self.nbr_p_matched, self.nbrP, self.nbrGT = 0, 0, 0, 0

        self.global_intrinsic = {
            "MLA": 0,
            "MSP": 0,
            "NE": 0,
            "NS": 0,
            "NSP": 0,
            "NTP": 0,
        }

        self.global_intrinsic_GT = {
            "MLA": 0,
            "MSP": 0,
            "NE": 0,
            "NS": 0,
            "NSP": 0,
            "NTP": 0,
        }

        self.global_edges = {
            "PCont": 0,
            "RCont": 0,
            "PCont_max": 0,
            "RCont_max": 0,
            "POri": 0,
            "ROri": 0,
            "POri_max": 0,
            "ROri_max": 0,
        }

        self.global_topo = {"TFP": 0, "TFN": 0, "OS": 0, "US": 0}

        self.global_surfacic = None

        self.global_3d = None

    def build_computers(self):

        for family in self.config.metrics_families:
            computer = None
            if family == "intrinsic":
                computer = pyscoring.metrics.IntrinsicMetrics(
                    self.dataLoader.project_crs
                )

            elif family == "topo":
                computer = pyscoring.metrics.Topological_IoMA(
                    threshold=self.config.threshold, extend_matches=self.config.extended
                )

            elif family == "edges":
                computer = pyscoring.metrics.EdgesMetric(
                    self.config.resolution, matched=True, union=True
                )

            elif family == "surfacic":
                computer = pyscoring.metrics.SurfacesMetric(
                    resolution=self.config.resolution
                )

            elif family == "3d":
                computer = pyscoring.metrics.ZMetric(
                    start_index=self.config.start_index,
                    epsg=self.config.epsg,
                    no_data=self.config.no_data,
                    save_raster=self.config.outdir if self.config.rasters else None,
                )

            self.computers.append([computer, family])

    def launch_parallelization(self):
        start_metrics = time.time()
        tiles_list = self.dataLoader.tiles
        i_3d = None
        for i in range(len(self.computers)):
            if self.computers[i][1] == "3d":
                i_3d = i
                break
        if i_3d is not None:
            computers = [] + self.computers[0:i_3d] + self.computers[i_3d + 1 :]
        else:
            computers = self.computers

        p_trsf = self.dataLoader.p_trsf
        gt_trsf = self.dataLoader.gt_trsf

        if len(computers) >= 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.nb_workers
            ) as executor:
                futures = {
                    executor.submit(computeMetrics, tile, computers, p_trsf, gt_trsf)
                    for tile in tiles_list
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Calculation of the metrics",
                ):
                    res = future.result()
                    self.output.append(res)

        # For now global surfacic is out of parallelization
        for c in self.computers:
            computer, family = c[0], c[1]
            if family == "surfacic" or family == "3d":
                start_mask = time.time()
                computer.reset()
                computer.update(
                    self.dataLoader.list_p,
                    self.dataLoader.list_gt,
                    rasterio.features.bounds(self.dataLoader.working_bbox),
                    self.config.resolution,
                )

                metrics = computer.compute()
                if family == "surfacic":
                    self.global_surfacic = metrics
                elif family == "3d":
                    self.global_3d = metrics

                end_mask = time.time()
                self.config.time(start_mask, end_mask, "Global mask generation")

        end_metrics = time.time()
        self.config.time(start_metrics, end_metrics, "Metrics calculation")

    def print_out(self):
        start_out = time.time()
        if self.config.mode == 3:
            MF = pyscoring.metrics.metric_merger.MergerFactory()
            merger = MF.create("multiple")

            for tile in self.output:
                self.global_topo = merger.merge_topo(
                    self.global_topo,
                    tile["topo"],
                    self.nbr_p_matched,
                    self.nbr_gt_matched,
                    self.nbrP,
                    self.nbrGT,
                )

                self.nbrP += tile["topo"][4]
                self.nbrGT += tile["topo"][5]

                if tile["topo"][4] > 0 and tile["topo"][5] > 0:
                    self.nbr_p_matched += tile["topo"][4]
                    self.nbr_gt_matched += tile["topo"][5]

            with open(self.config.outpath_total, "w", newline="") as file:
                writer = csv.writer(file)

                list_to_write = [
                    [
                        "Match_Category",
                        "Number predictions",
                        "Number GT",
                        "Number of matched P",
                        "Number of matched GT",
                        "False Positive Rate",
                        "False Negative Rate",
                        "Over Segmentation",
                        "Under Segmentation",
                        "Under reconstruction rate (pixels)",
                        "Over reconstruction rate (pixels)",
                        "Sum error",
                        "Standard deviation",
                    ],
                    [
                        "ALL",
                        self.nbrP,
                        self.nbrGT,
                        self.nbr_p_matched,
                        self.nbr_gt_matched,
                    ]
                    + list(self.global_topo.values())
                    + list(self.global_3d),
                ]

                for values in list_to_write:
                    writer.writerow(values)

        else:

            # Call metrics mergers
            MF = pyscoring.metrics.metric_merger.MergerFactory()
            merge_one_one = MF.create("multiple")
            merge_n_one = MF.create("multiple")
            merge_one_m = MF.create("multiple")
            merge_n_m = MF.create("multiple")
            merge_one_o = MF.create("one_o")
            merge_o_one = MF.create("o_one")

            list_mergers = [
                [merge_one_one, "1-1"],
                [merge_n_one, "N-1"],
                [merge_one_m, "1-M"],
                [merge_n_m, "N-M"],
                [merge_one_o, "1-0"],
                [merge_o_one, "0-1"],
            ]

            # Print out the metrics building by building, and merge them in global metrics
            with open(self.config.outpath_building, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Cluster of buildings",
                        "id_P",
                        "id_GT",
                        "Matched",
                        "Nbr_P",
                        "Nbr_GT",
                        "OverSeg",
                        "UnderSeg",
                        "Dist_P",
                        "Dist_GT",
                        "Distmax_P",
                        "Distmax_GT",
                        "Ori_P",
                        "Ori_GT",
                        "Orimax_P",
                        "Orimax_GT",
                        "Surf_TP",
                        "Surf_TN",
                        "Surf_FP",
                        "Surf_FN",
                        "Surf_Pr",
                        "Surf_Ra",
                        "LenEdge_P",
                        "Surface_P",
                        "NbrPts_P",
                        "MeanPts_P",
                        "MeanPit_P",
                        "LenEdge_GT",
                        "Surface_GT",
                        "NbrPts_GT",
                        "MeanPts_GT",
                        "MeanPit_GT",
                    ]
                )

                schema = {
                    "geometry": "Polygon",
                    "properties": {
                        "id_P": "str",
                        "id_GT": "str",
                        "Match": "str",
                        "Nbr_P": "int",
                        "Nbr_GT": "int",
                        "OverSeg": "float",
                        "UnderSeg": "float",
                        "Dist_P": "float",
                        "Dist_GT": "float",
                        "Distmax_P": "float",
                        "Distmax_GT": "float",
                        "Ori_P": "float",
                        "Ori_GT": "float",
                        "Orimax_P": "float",
                        "Orimax_GT": "float",
                        "Surf_TP": "float",
                        "Surf_TN": "float",
                        "Surf_FP": "float",
                        "Surf_FN": "float",
                        "Surf_Pr": "float",
                        "Surf_Ra": "float",
                        "LenEdge_P": "float",
                        "Surface_P": "float",
                        "NbrPts_P": "int",
                        "MeanPts_P": "float",
                        "MeanPit_P": "float",
                        "LenEdge_GT": "float",
                        "Surface_GT": "float",
                        "NbrPts_GT": "int",
                        "MeanPts_GT": "float",
                        "MeanPit_GT": "float",
                    },
                }

                with fiona.open(
                    self.config.outpath_building[:-3] + "shp",
                    "w",
                    "ESRI Shapefile",
                    schema,
                ) as shp:
                    output = self.output
                    for tile in output:
                        if (
                            list(tile["intrinsic"])[2]
                            == list(tile["intrinsic_GT"])[2]
                            == 1
                        ):
                            merger = merge_one_one
                        elif list(tile["intrinsic"])[2] == 0:
                            merger = merge_o_one
                        elif list(tile["intrinsic_GT"])[2] == 0:
                            merger = merge_one_o
                        elif list(tile["intrinsic"])[2] == 1:
                            merger = merge_one_m
                        elif list(tile["intrinsic_GT"])[2] == 1:
                            merger = merge_n_one
                        else:
                            merger = merge_n_m

                        out = [
                            wkt.dumps(tile["cluster"]),
                            tile["idp"],
                            tile["idg"],
                            tile["matched"],
                        ]
                        out = out + [
                            list(tile["intrinsic"])[2],
                            list(tile["intrinsic_GT"])[2],
                        ]
                        out = out + [
                            "{:.2f}".format(value) for value in list(tile["topo"])[2:-2]
                        ]
                        out = out + [
                            "{:.2f}".format(value) for value in list(tile["edges"])[:-2]
                        ]
                        out = out + [
                            "{:.2f}".format(value)
                            for value in list(tile["surfacic"])[:-1]
                        ]
                        out = (
                            out
                            + [
                                "{:.2f}".format(float(value))
                                for value in list(tile["intrinsic"])[:2]
                            ]
                            + [list(tile["intrinsic"])[3]]
                            + [
                                "{:.2f}".format(float(value))
                                for value in list(tile["intrinsic"])[4:]
                            ]
                        )
                        out = (
                            out
                            + [
                                "{:.2f}".format(float(value))
                                for value in list(tile["intrinsic_GT"])[:2]
                            ]
                            + [list(tile["intrinsic_GT"])[3]]
                            + [
                                "{:.2f}".format(float(value))
                                for value in list(tile["intrinsic_GT"])[4:]
                            ]
                        )

                        writer.writerow(out)
                        shp.write(
                            {
                                "geometry": mapping(tile["cluster"]),
                                "properties": {
                                    "id_P": tile["idp"],
                                    "id_GT": tile["idg"],
                                    "Match": tile["matched"],
                                    "Nbr_P": list(tile["intrinsic"])[2],
                                    "Nbr_GT": list(tile["intrinsic_GT"])[2],
                                    "OverSeg": "{:.2f}".format(list(tile["topo"])[3]),
                                    "UnderSeg": "{:.2f}".format(list(tile["topo"])[4]),
                                    "Dist_P": "{:.2f}".format(list(tile["edges"])[0]),
                                    "Dist_GT": "{:.2f}".format(list(tile["edges"])[1]),
                                    "Distmax_P": "{:.2f}".format(
                                        list(tile["edges"])[2]
                                    ),
                                    "Distmax_GT": "{:.2f}".format(
                                        list(tile["edges"])[3]
                                    ),
                                    "Ori_P": "{:.2f}".format(list(tile["edges"])[4]),
                                    "Ori_GT": "{:.2f}".format(list(tile["edges"])[5]),
                                    "Orimax_P": "{:.2f}".format(list(tile["edges"])[6]),
                                    "Orimax_GT": "{:.2f}".format(
                                        list(tile["edges"])[7]
                                    ),
                                    "Surf_TP": "{:.2f}".format(
                                        list(tile["surfacic"])[0]
                                    ),
                                    "Surf_TN": "{:.2f}".format(
                                        list(tile["surfacic"])[1]
                                    ),
                                    "Surf_FP": "{:.2f}".format(
                                        list(tile["surfacic"])[2]
                                    ),
                                    "Surf_FN": "{:.2f}".format(
                                        list(tile["surfacic"])[3]
                                    ),
                                    "Surf_Pr": "{:.2f}".format(
                                        list(tile["surfacic"])[4]
                                    ),
                                    "Surf_Ra": "{:.2f}".format(
                                        list(tile["surfacic"])[5]
                                    ),
                                    "LenEdge_P": "{:.2f}".format(
                                        list(tile["intrinsic"])[0]
                                    ),
                                    "Surface_P": "{:.2f}".format(
                                        list(tile["intrinsic"])[1]
                                    ),
                                    "NbrPts_P": list(tile["intrinsic"])[3],
                                    "MeanPts_P": "{:.2f}".format(
                                        list(tile["intrinsic"])[4]
                                    ),
                                    "MeanPit_P": "{:.2f}".format(
                                        list(tile["intrinsic"])[5]
                                    ),
                                    "LenEdge_GT": "{:.2f}".format(
                                        list(tile["intrinsic_GT"])[0]
                                    ),
                                    "Surface_GT": "{:.2f}".format(
                                        list(tile["intrinsic_GT"])[1]
                                    ),
                                    "NbrPts_GT": list(tile["intrinsic_GT"])[3],
                                    "MeanPts_GT": "{:.2f}".format(
                                        list(tile["intrinsic_GT"])[4]
                                    ),
                                    "MeanPit_GT": "{:.2f}".format(
                                        list(tile["intrinsic_GT"])[5]
                                    ),
                                },
                            }
                        )

                        (
                            self.global_intrinsic,
                            self.global_intrinsic_GT,
                            self.global_edges,
                            self.global_topo,
                            self.nbr_p_matched,
                            self.nbr_gt_matched,
                            self.nbrP,
                            self.nbrGT,
                        ) = merger.merge(
                            tile,
                            self.global_intrinsic,
                            self.global_intrinsic_GT,
                            self.global_edges,
                            self.global_topo,
                            self.nbr_p_matched,
                            self.nbr_gt_matched,
                            self.nbrP,
                            self.nbrGT,
                        )

            # Print out the global merged metrics

            with open(self.config.outpath_total, "w", newline="") as file:
                writer = csv.writer(file)

                list_to_write = [
                    [
                        "Match_Category",
                        "Nombre de prédictions",
                        "Nombre de GT",
                        "Nombre de P matchés",
                        "Nombre de GT matchés",
                        "Taux de Faux Positifs",
                        "Taux de Faux Négatifs",
                        "Sur Segmentation",
                        "Sous Segmentation",
                        "Moyenne Longueur Arrêtes",
                        "Moy. Surface/Polygone",
                        "Nombre d'emprises",
                        "Nombre de sommets",
                        "Nombre Sommets/Polyg",
                        "Nombre Trou/Polyg",
                        "Moyenne Longueur Arrêtes GT",
                        "Moy. Surface/Polygone GT",
                        "Nombre d'emprises GT",
                        "Nombre de sommets GT",
                        "Nombre Sommets/Polyg GT",
                        "Nombre Trou/Polyg GT",
                        "Precision de Contour",
                        "Rappel de Contour",
                        "Precision Contour max",
                        "Rappel Contour max",
                        "Précision d'Orientation",
                        "Rappel d'Orientation",
                        "Precision Orientation max",
                        "Rappel Orientation max",
                        "Taux vrais positifs surf.",
                        "Taux vrais negatifs surf.",
                        "Taux faux positifs surf.",
                        "Taux faux negatifs surf.",
                        "Precision",
                        "Rappel",
                    ],
                    [
                        "ALL",
                        len(self.dataLoader.list_p),
                        len(self.dataLoader.list_gt),
                        self.nbr_p_matched,
                        self.nbr_gt_matched,
                    ]
                    + list(self.global_topo.values())
                    + list(self.global_intrinsic.values())
                    + list(self.global_intrinsic_GT.values())
                    + list(self.global_edges.values())
                    + list(self.global_surfacic)[:-1],
                ]

                for merger in list_mergers:
                    # Add all the metrics for this specific c category of match
                    m = merger[0]
                    label = merger[1]

                    (
                        c_intrinsic,
                        c_intrinsic_gt,
                        c_edges,
                        c_topo,
                        c_nbr_p_matched,
                        c_nbr_gt_matched,
                        c_nbrP,
                        c_nbrGT,
                    ) = m.get_self_metrics()
                    output = (
                        [label, c_nbrP, c_nbrGT, c_nbr_p_matched, c_nbr_gt_matched]
                        + list(c_topo.values())
                        + list(c_intrinsic.values())
                        + list(c_intrinsic_gt.values())
                        + list(c_edges.values())
                        + ["-" in range(6)]
                    )

                    list_to_write.append(output)

                for values in list_to_write:
                    writer.writerow(values)

        end_out = time.time()
        self.config.time(start_out, end_out, "Merge and print out")


def computeMetrics(tile, computers, p_trsf, gt_trsf):
    """
    :param tile: a tile is a list of polygons_p, polygons_gt, id_p, id_gt
    :return:
    """
    polygons_p, polygons_gt, id_p, id_gt = tile

    metrics = {}  # output dictionary
    if len(polygons_gt) == 1 and len(polygons_p) == 1:
        metrics["matched"] = "1-1"
    elif len(polygons_gt) == 0:
        metrics["matched"] = "1-0"
    elif len(polygons_p) == 0:
        metrics["matched"] = "0-1"
    elif len(polygons_gt) == 1:
        metrics["matched"] = "N-1"
    elif len(polygons_p) == 1:
        metrics["matched"] = "1-M"
    else:
        metrics["matched"] = "N-M"

    for metric_computer, metrics_family in computers:
        if metrics_family == "intrinsic":
            metric_computer.update(polygons_p)
            metrics[metrics_family] = metric_computer.compute()
            metric_computer.reset()
            metric_computer.update(polygons_gt)
            metrics[metrics_family + "_GT"] = metric_computer.compute()

        else:
            metric_computer.update(polygons_p, polygons_gt)
            metrics[metrics_family] = metric_computer.compute()

    # To represent the metrics choose a geometry
    try:
        cluster_polygon = shapely.unary_union(polygons_p + polygons_gt)
        if p_trsf:
            cluster_polygon = transform(p_trsf, cluster_polygon)
        elif gt_trsf:
            cluster_polygon = transform(gt_trsf, cluster_polygon)

    except GEOSException:
        print("Invalid polygon:" + str(polygons_p))
        cluster_polygon = polygons_p[0] if len(polygons_p) > 0 else polygons_gt[0]

    metrics["cluster"] = cluster_polygon
    metrics["idp"] = ", ".join(map(str, id_p))
    metrics["idg"] = ", ".join(map(str, id_gt))

    return metrics
