class Merger:
    r"""Class to merge the metrics produced in different patches

    Notes:
        Every family of metrics need different merging :

        * Edges metrics are merged with the merge_edges function
        * Intrinsic metrics are merged with the merge_intr function
        * Topological metrics are merged with the merge_topo function

        Every case of match also requires different merging, made within the classes:

        * (NxM) matches, with N and M >= 1 polygon, are merged by the Merger_multiple class
        * (0x1) matches are merged by the Merger_o_one class
        * (1x0) matches are merged by the Merger_one_o class

        Every class should be built by the merger factory
    """

    def __init__(self):
        """
        Init the merger

        Notes :
            * Initiates to the basic value every metric of every family of metrics.
            * Each family has its own dictionary : intrinsics, intrinsics_GT, edges, topological
            * Keeping the statistics of the polygons already met with nbr_gt_matched, nbr_p_matched, nbrP, nbrGT

            * Metrics :
                * intrinsics & intrinsics_GT : MLA mean edge's length, MSP mean polygon's surface, NE number of polygons
                , NS number of vertex, NSP mean vertexes by polygon, NTP mean holes by polygons

                * edges : PCont distance pred->gt, RCont distance g->p, PCont_max/RCont_max max distance p->g/g->p,
                POri orientation diff p->g, ROri orientation diff g->p, POri_max/ROri_max max ori diff p->g/g->p

                * topo : TFP false positives rate, TFN false negative rate, OS over segmentation rate,
                 US under segmentation rate
        """
        self.by_type = True

        self.intrinsics = {"MLA": 0, "MSP": 0, "NE": 0, "NS": 0, "NSP": 0, "NTP": 0}

        self.intrinsics_GT = {"MLA": 0, "MSP": 0, "NE": 0, "NS": 0, "NSP": 0, "NTP": 0}

        self.edges = {
            "PCont": 0,
            "RCont": 0,
            "PCont_max": 0,
            "RCont_max": 0,
            "POri": 0,
            "ROri": 0,
            "POri_max": 0,
            "ROri_max": 0,
        }

        self.topo = {"TFP": 0, "TFN": 0, "OS": 0, "US": 0}

        self.nbr_gt_matched = 0
        self.nbr_p_matched = 0
        self.nbrP = 0
        self.nbrGT = 0

    def set_by_type(self, by_type: bool):
        self.by_type = by_type

    def merge(
        self,
        tile_metrics,
        intrinsics,
        intrinsics_gt,
        edges,
        topo,
        nbr_p_matched,
        nbr_gt_matched,
        nbrP,
        nbrGT,
    ) -> None:
        """Override this method to correctly merge the metrics in each case of match"""
        raise NotImplementedError

    def merge_edges(self, edges, intrinsics, local_edg, local_intr):
        """
        Merge the edges metrics locally

        :param edges:
        :param intrinsics:
        :param local_edg:
        :param local_intr:
        :return:
        """
        # Merge the edges metrics : global
        edges["RCont"] = (
            edges["RCont"] * intrinsics["MLA"] * intrinsics["NS"]
            + local_edg[1] * local_intr[0] * local_intr[3]
        ) / (intrinsics["MLA"] * intrinsics["NS"] + local_intr[0] * local_intr[3])
        edges["ROri"] = (
            edges["ROri"] * intrinsics["MLA"] * intrinsics["NS"]
            + local_edg[5] * local_intr[0] * local_intr[3]
        ) / (intrinsics["MLA"] * intrinsics["NS"] + local_intr[0] * local_intr[3])
        edges["PCont"] = (
            edges["PCont"] * intrinsics["MLA"] * intrinsics["NS"]
            + local_edg[0] * local_intr[0] * local_intr[3]
        ) / (intrinsics["MLA"] * intrinsics["NS"] + local_intr[0] * local_intr[3])
        edges["POri"] = (
            edges["POri"] * intrinsics["MLA"] * intrinsics["NS"]
            + local_edg[4] * local_intr[0] * local_intr[3]
        ) / (intrinsics["MLA"] * intrinsics["NS"] + local_intr[0] * local_intr[3])

        edges["PCont_max"] = (
            local_edg[2] if local_edg[2] > edges["PCont_max"] else edges["PCont_max"]
        )
        edges["RCont_max"] = (
            local_edg[3] if local_edg[3] > edges["RCont_max"] else edges["RCont_max"]
        )
        edges["POri_max"] = (
            local_edg[6] if local_edg[6] > edges["POri_max"] else edges["POri_max"]
        )
        edges["ROri_max"] = (
            local_edg[7] if local_edg[7] > edges["ROri_max"] else edges["ROri_max"]
        )

        # Merge the edges metrics : type
        if self.by_type:
            self.merge_edges_self(local_edg, local_intr)

        return edges

    def merge_intr(self, intrinsics, local_intr, gt: bool = False):
        """
        Merge the intrinsic metrics locally

        :param intrinsics:
        :param local_intr:
        :param gt:
        :return:
        """
        # Merge the intrinsic metrics : global
        intrinsics["MLA"] = (
            intrinsics["MLA"] * intrinsics["NS"] + local_intr[0] * local_intr[3]
        ) / (intrinsics["NS"] + local_intr[3])
        intrinsics["MSP"] = (
            intrinsics["MSP"] * intrinsics["NE"] + local_intr[1] * local_intr[2]
        ) / (intrinsics["NE"] + local_intr[2])
        intrinsics["NSP"] = (
            intrinsics["NSP"] * intrinsics["NE"] + local_intr[4] * local_intr[2]
        ) / (intrinsics["NE"] + local_intr[2])
        intrinsics["NTP"] = (
            intrinsics["NTP"] * intrinsics["NE"] + local_intr[5] * local_intr[2]
        ) / (intrinsics["NE"] + local_intr[2])
        intrinsics["NE"] += local_intr[2]
        intrinsics["NS"] += local_intr[3]

        # Merge the intrinsic metrics : type
        if self.by_type:
            self.merge_intr_self(local_intr, gt)

        return intrinsics

    def merge_topo(self, topo, local_topo, nbr_p_matched, nbr_gt_matched, nbrP, nbrGT):
        """
        Merge the topological metrics locally
        :param topo:
        :param local_topo:
        :param nbr_p_matched:
        :param nbr_gt_matched:
        :param nbrP:
        :param nbrGT:
        :return:
        """
        if local_topo[4] != 0:
            topo["TFP"] = (topo["TFP"] * nbrP + local_topo[0] * local_topo[4]) / (
                nbrP + local_topo[4]
            )
        if local_topo[5] != 0:
            topo["TFN"] = (topo["TFN"] * nbrGT + local_topo[1] * local_topo[5]) / (
                nbrGT + local_topo[5]
            )
        if (
            local_topo[4] != 0
            and local_topo[5] != 0
            and local_topo[0] != 1
            and local_topo[1] != 1
        ):
            topo["OS"] = (
                topo["OS"] * nbr_gt_matched
                + local_topo[2] * ((1 - local_topo[1]) * local_topo[5])
            ) / (nbr_gt_matched + ((1 - local_topo[1]) * local_topo[5]))
            topo["US"] = (
                topo["US"] * nbr_p_matched
                + local_topo[3] * ((1 - local_topo[0]) * local_topo[4])
            ) / (nbr_p_matched + ((1 - local_topo[0]) * local_topo[4]))

        if self.by_type:
            self.merge_topo_self(local_topo)

        return topo

    def get_self_metrics(self):
        return (
            self.intrinsics,
            self.intrinsics_GT,
            self.edges,
            self.topo,
            self.nbr_p_matched,
            self.nbr_gt_matched,
            self.nbrP,
            self.nbrGT,
        )

    def merge_edges_self(self, local_edg, local_intr):
        """
        Merge the local edges metrics with the already saved edges metrics to get the updated global ones
        :param local_edg:
        :param local_intr:
        :return:
        """
        self.edges["RCont"] = (
            self.edges["RCont"] * self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_edg[1] * local_intr[0] * local_intr[3]
        ) / (
            self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_intr[0] * local_intr[3]
        )
        self.edges["ROri"] = (
            self.edges["ROri"] * self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_edg[5] * local_intr[0] * local_intr[3]
        ) / (
            self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_intr[0] * local_intr[3]
        )
        self.edges["PCont"] = (
            self.edges["PCont"] * self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_edg[0] * local_intr[0] * local_intr[3]
        ) / (
            self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_intr[0] * local_intr[3]
        )
        self.edges["POri"] = (
            self.edges["POri"] * self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_edg[4] * local_intr[0] * local_intr[3]
        ) / (
            self.intrinsics["MLA"] * self.intrinsics["NS"]
            + local_intr[0] * local_intr[3]
        )

        self.edges["PCont_max"] = (
            local_edg[2]
            if local_edg[2] > self.edges["PCont_max"]
            else self.edges["PCont_max"]
        )
        self.edges["RCont_max"] = (
            local_edg[3]
            if local_edg[3] > self.edges["RCont_max"]
            else self.edges["RCont_max"]
        )
        self.edges["POri_max"] = (
            local_edg[6]
            if local_edg[6] > self.edges["POri_max"]
            else self.edges["POri_max"]
        )
        self.edges["ROri_max"] = (
            local_edg[7]
            if local_edg[7] > self.edges["ROri_max"]
            else self.edges["ROri_max"]
        )

    def merge_intr_self(self, local_intr, gt=False):
        """
        Merge the local intrinsic metrics with the already saved intrinsic metrics to get the updated global ones
        :param local_intr:
        :param gt:
        :return:
        """
        if gt:
            self.intrinsics_GT["MLA"] = (
                self.intrinsics_GT["MLA"] * self.intrinsics_GT["NS"]
                + local_intr[0] * local_intr[3]
            ) / (self.intrinsics_GT["NS"] + local_intr[3])
            self.intrinsics_GT["MSP"] = (
                self.intrinsics_GT["MSP"] * self.intrinsics_GT["NE"]
                + local_intr[1] * local_intr[2]
            ) / (self.intrinsics_GT["NE"] + local_intr[2])
            self.intrinsics_GT["NSP"] = (
                self.intrinsics_GT["NSP"] * self.intrinsics_GT["NE"]
                + local_intr[4] * local_intr[2]
            ) / (self.intrinsics_GT["NE"] + local_intr[2])
            self.intrinsics_GT["NTP"] = (
                self.intrinsics_GT["NTP"] * self.intrinsics_GT["NE"]
                + local_intr[5] * local_intr[2]
            ) / (self.intrinsics_GT["NE"] + local_intr[2])
            self.intrinsics_GT["NE"] += local_intr[2]
            self.intrinsics_GT["NS"] += local_intr[3]
        else:
            self.intrinsics["MLA"] = (
                self.intrinsics["MLA"] * self.intrinsics["NS"]
                + local_intr[0] * local_intr[3]
            ) / (self.intrinsics["NS"] + local_intr[3])
            self.intrinsics["MSP"] = (
                self.intrinsics["MSP"] * self.intrinsics["NE"]
                + local_intr[1] * local_intr[2]
            ) / (self.intrinsics["NE"] + local_intr[2])
            self.intrinsics["NSP"] = (
                self.intrinsics["NSP"] * self.intrinsics["NE"]
                + local_intr[4] * local_intr[2]
            ) / (self.intrinsics["NE"] + local_intr[2])
            self.intrinsics["NTP"] = (
                self.intrinsics["NTP"] * self.intrinsics["NE"]
                + local_intr[5] * local_intr[2]
            ) / (self.intrinsics["NE"] + local_intr[2])
            self.intrinsics["NE"] += local_intr[2]
            self.intrinsics["NS"] += local_intr[3]

    def merge_topo_self(self, local_topo):
        """
        Merge the local topological metrics with the already saved topological metrics to get the updated global ones
        :param local_topo:
        :return:
        """
        if local_topo[4] != 0:
            self.topo["TFP"] = (
                self.topo["TFP"] * self.nbrP + local_topo[0] * local_topo[4]
            ) / (self.nbrP + local_topo[4])
        if local_topo[5] != 0:
            self.topo["TFN"] = (
                self.topo["TFN"] * self.nbrGT + local_topo[1] * local_topo[5]
            ) / (self.nbrGT + local_topo[5])
        if (
            local_topo[4] != 0
            and local_topo[5] != 0
            and local_topo[0] != 1
            and local_topo[1] != 1
        ):
            self.topo["OS"] = (
                self.topo["OS"] * self.nbr_gt_matched
                + local_topo[2] * ((1 - local_topo[1]) * local_topo[5])
            ) / (self.nbr_gt_matched + ((1 - local_topo[1]) * local_topo[5]))
            self.topo["US"] = (
                self.topo["US"] * self.nbr_p_matched
                + local_topo[3] * ((1 - local_topo[0]) * local_topo[4])
            ) / (self.nbr_p_matched + ((1 - local_topo[0]) * local_topo[4]))


class Merger_multiple(Merger):
    """
    Class for the case of the multiple match (NxM)
    """

    def __init__(self):
        super().__init__()

    def merge(
        self,
        tile_metrics,
        intrinsics,
        intrinsics_gt,
        edges,
        topo,
        nbr_p_matched,
        nbr_gt_matched,
        nbrP,
        nbrGT,
    ):
        local_intr = list(tile_metrics["intrinsic"])
        local_intr_gt = list(tile_metrics["intrinsic_GT"])
        local_edg = list(tile_metrics["edges"])
        local_topo = list(tile_metrics["topo"])

        edges = self.merge_edges(edges, intrinsics, local_edg, local_intr)
        intrinsics = self.merge_intr(intrinsics, local_intr)
        intrinsics_gt = self.merge_intr(intrinsics_gt, local_intr_gt, True)
        topo = self.merge_topo(
            topo, local_topo, nbr_p_matched, nbr_gt_matched, nbrP, nbrGT
        )

        nbr_p_matched += (1 - local_topo[0]) * local_topo[4]
        nbr_gt_matched += (1 - local_topo[1]) * local_topo[5]

        nbrP += local_intr[2]
        nbrGT += local_intr_gt[2]

        if self.by_type:
            self.nbr_p_matched += (1 - local_topo[0]) * local_topo[4]
            self.nbr_gt_matched += (1 - local_topo[1]) * local_topo[5]

            self.nbrP += local_intr[2]
            self.nbrGT += local_intr_gt[2]

        return (
            intrinsics,
            intrinsics_gt,
            edges,
            topo,
            nbr_p_matched,
            nbr_gt_matched,
            nbrP,
            nbrGT,
        )


class Merger_one_o(Merger):
    """
    Class for the case of the non-match (1x0)
    """

    def __init__(self):
        super().__init__()

    def merge(
        self,
        tile_metrics,
        intrinsics,
        intrinsics_gt,
        edges,
        topo,
        nbr_p_matched,
        nbr_gt_matched,
        nbrP,
        nbrGT,
    ):
        local_intr = list(tile_metrics["intrinsic"])
        local_intr_gt = list(tile_metrics["intrinsic_GT"])
        local_topo = list(tile_metrics["topo"])

        intrinsics = self.merge_intr(intrinsics, local_intr)
        topo = self.merge_topo(
            topo, local_topo, nbr_p_matched, nbr_gt_matched, nbrP, nbrGT
        )

        nbrP += local_intr[2]
        nbrGT += local_intr_gt[2]

        if self.by_type:
            self.nbrP += local_intr[2]
            self.nbrGT += local_intr_gt[2]

        return (
            intrinsics,
            intrinsics_gt,
            edges,
            topo,
            nbr_p_matched,
            nbr_gt_matched,
            nbrP,
            nbrGT,
        )


class Merger_o_one(Merger):
    """
    Class for the case of the non-match (0x1)
    """

    def merge(
        self,
        tile_metrics,
        intrinsics,
        intrinsics_gt,
        edges,
        topo,
        nbr_p_matched,
        nbr_gt_matched,
        nbrP,
        nbrGT,
    ):
        local_intr = list(tile_metrics["intrinsic"])
        local_intr_gt = list(tile_metrics["intrinsic_GT"])
        local_topo = list(tile_metrics["topo"])

        intrinsics_gt = self.merge_intr(intrinsics_gt, local_intr_gt, True)
        topo = self.merge_topo(
            topo, local_topo, nbr_p_matched, nbr_gt_matched, nbrP, nbrGT
        )

        nbrP += local_intr[2]
        nbrGT += local_intr_gt[2]

        if self.by_type:
            self.nbrP += local_intr[2]
            self.nbrGT += local_intr_gt[2]

        return (
            intrinsics,
            intrinsics_gt,
            edges,
            topo,
            nbr_p_matched,
            nbr_gt_matched,
            nbrP,
            nbrGT,
        )


class MergerFactory:
    """ "
    Instantiate the different metrics class
    """

    def __init__(self):
        None

    def create(self, match_type):
        matcher_type = "Merger_" + match_type
        return globals()[matcher_type]()
