import time
from math import floor, ceil

from cjio import cityjson
from shapely.geometry import Polygon, LineString, mapping
import geopandas as gpd
import pyproj
import sys
import rasterio.features
import rasterio.features
import shapely
from shapely import intersection, GEOSException, force_2d, make_valid, union_all

from rtree import index
import networkx as nx
from shapely.errors import TopologicalError
from shapely.validation import explain_validity

import pyscoring


class DataLoader:
    def __init__(self, conf_pyscoring):
        # Data
        self.config = conf_pyscoring
        self.working_bbox = None
        self.project_crs = None

        # GT
        self.gt_surfaces = None
        self.gt_global_bbox = None
        self.gt_crs = None
        self.gt_trsf = None
        self.list_gt = []

        # P
        self.p_surfaces = None
        self.p_global_bbox = None
        self.p_crs = None
        self.p_trsf = None
        self.list_p = []

        if self.config.mode == 3:
            self.gt_roof_polygons_3d = None
            self.p_roof_polygons_3d = None
            self.gt_ground_2d_global_bbox = None
            self.p_ground_2d_global_bbox = None

        # Parallelization tiles
        self.matches_graph = None
        self.tiles = None

    def load_data(self):
        if self.config.mode == 2:
            self.load_data_2d()
        else:
            self.load_data_3d()

    def load_data_3d(self):
        self.gt_roof_polygons_3d, self.gt_ground_2d_global_bbox = self.get_surfaces(
            self.config.gt_path
        )
        self.p_roof_polygons_3d, self.p_ground_2d_global_bbox = self.get_surfaces(
            self.config.p_path
        )

        working_bbox = list(
            merge_bounds(
                tuple(self.gt_ground_2d_global_bbox),
                tuple(self.gt_ground_2d_global_bbox),
                self.config.resolution,
            )
        )

        self.working_bbox = Polygon(
            [
                (working_bbox[0], working_bbox[1]),
                (working_bbox[0], working_bbox[3]),
                (working_bbox[2], working_bbox[3]),
                (working_bbox[2], working_bbox[1]),
            ]
        )

    def load_data_2d(self):
        start_load = time.time()

        self.gt_surfaces, self.gt_global_bbox, self.gt_crs, self.gt_trsf = (
            self.get_surfaces(
                self.config.gt_path, self.config.gt_class, self.config.epsg
            )
        )
        self.p_surfaces, self.p_global_bbox, self.p_crs, self.p_trsf = (
            self.get_surfaces(self.config.p_path, self.config.p_class, self.config.epsg)
        )

        # Test if same CRS :
        if self.gt_crs != self.p_crs:
            self.config.pys_print(
                "Warning the data CRS are not the same, please correct this."
            )
            sys.exit(0)
        else:
            self.project_crs = self.gt_crs

        # Construction of the working bbox
        if not self.config.w_bbox_source:
            # If nothing put then build the working_bbox from the input data
            working_bbox = list(
                merge_bounds(
                    self.gt_global_bbox, self.p_global_bbox, self.config.resolution
                )
            )

            self.working_bbox = Polygon(
                [
                    (working_bbox[0], working_bbox[1]),
                    (working_bbox[0], working_bbox[3]),
                    (working_bbox[2], working_bbox[3]),
                    (working_bbox[2], working_bbox[1]),
                ]
            )
        else:
            try:
                self.working_bbox = self.get_surfaces(
                    self.config.w_bbox_source, None, self.config.epsg
                )[0]
            except Exception as e:
                self.config.pys_print(
                    "Not correct file of bbox, will use mixed p-gt bbox : {}".format(e)
                )
                working_bbox = list(
                    merge_bounds(
                        self.gt_global_bbox, self.p_global_bbox, self.config.resolution
                    )
                )
                self.working_bbox = Polygon(
                    [
                        (working_bbox[0], working_bbox[1]),
                        (working_bbox[0], working_bbox[3]),
                        (working_bbox[2], working_bbox[3]),
                        (working_bbox[2], working_bbox[1]),
                    ]
                )

        # If necessary, exclusion of the surfaces out of the boundaries of the zone
        self.gt_surfaces = exclusion(
            self.gt_surfaces, self.working_bbox, self.config.strict_bool
        )
        self.p_surfaces = exclusion(
            self.p_surfaces, self.working_bbox, self.config.strict_bool
        )

        end_load = time.time()
        self.config.time(start_load, end_load, "Load data")

    def tiling(self):
        start_tiling = time.time()
        self.build_graph()
        self.tiles = []  # init the list of tiles

        for connected_comp in list(nx.connected_components(self.matches_graph)):
            tile = [
                [],
                [],
                [],
                [],
            ]  # one tile : list of p, list of gt, list p id, list gt id

            for i_node in list(connected_comp):  # node by node

                nature = self.matches_graph.nodes[i_node]["nature"]

                if nature == "gt":
                    feature = (
                        self.list_gt[i_node]
                        if self.config.mode == 2
                        else self.list_gt[i_node][0]
                    )
                    tile[1].append(feature)
                    id_data = (
                        None
                        if self.config.gt_id is None
                        else self.matches_graph.nodes[i_node]["obj"]
                    )
                    tile[3].append(id_data)
                else:
                    i_obj = i_node - len(self.list_gt)
                    feature = (
                        self.list_p[i_obj]
                        if self.config.mode == 2
                        else self.list_p[i_obj][0]
                    )
                    tile[0].append(feature)
                    id_data = (
                        None
                        if self.config.p_id is None
                        else self.matches_graph.nodes[i_node]["obj"]
                    )
                    tile[2].append(id_data)

            self.tiles.append(tile)
        end_tiling = time.time()
        self.config.time(start_tiling, end_tiling, "Matching and tiling")

    def build_graph(self):

        # Building the tiling around the matches gt-p
        if self.config.nb_workers is not None:
            if self.config.nb_workers <= 1:
                self.config.pys_print(
                    "Warning : only one or less workers but you are tiling for parallelization"
                )

        # Building polygons matching

        # create index for faster intersection
        spatial_index = index.Index()
        # create graph linking matched roof polygons
        graph = nx.Graph()

        if self.config.mode == 2:
            list_p, list_gt, graph = self.build_graph_2d(spatial_index, graph)
        else:
            list_p, list_gt, graph = self.build_graph_3d(spatial_index, graph)

        self.list_p = list_p
        self.list_gt = list_gt
        self.matches_graph = graph

    def build_graph_3d(self, spatial_index, graph):
        p_roofs, gt_roofs = [], []

        # building is string from the data
        for building_id, gt_roof_polygons in self.gt_roof_polygons_3d.items():
            valid_gt_roof_polygons, new_polygons_count = self.make_polygons_valid(
                gt_roof_polygons
            )
            # new_gt_count += new_polygons_count
            for gt_roof in valid_gt_roof_polygons:
                gt_i = len(gt_roofs)
                iraster = self.config.start_index + gt_i
                # why not shapely .bounds ?
                bbox = rasterio.features.bounds(gt_roof)
                spatial_index.insert(id=gt_i, coordinates=bbox)
                graph.add_node(gt_i, nature="gt", obj=building_id)
                gt_roofs.append((gt_roof, iraster))

        for building_id, p_roof_polygons in self.p_roof_polygons_3d.items():
            valid_p_roof_polygons, new_polygons_count = self.make_polygons_valid(
                p_roof_polygons
            )
            # new_p_count += new_polygons_count
            for p_roof in valid_p_roof_polygons:
                p_i = len(gt_roofs) + len(p_roofs)
                iraster = self.config.start_index + p_i
                bbox_pred = rasterio.features.bounds(p_roof)
                p_roofs.append((p_roof, iraster))
                p_roof_area = p_roof.area
                if p_roof_area == 0:
                    self.config.pys_print(
                        "Warning : area of Prediction "
                        + str(building_id)
                        + " is null, polygon ignored."
                    )
                    continue
                graph.add_node(p_i, nature="p", obj=building_id)
                # Match roof polygon with GT
                for igt_idx in spatial_index.intersection(bbox_pred):
                    gt_roof = gt_roofs[igt_idx][0]
                    gt_roof_area = gt_roof.area
                    if gt_roof_area == 0:
                        self.config.pys_print(
                            "Warning : area of GT "
                            + str(building_id)
                            + " is null, polygon ignored."
                        )
                        continue
                    try:
                        inter = intersection(p_roof, gt_roof)
                    except:
                        self.config.pys_print("Could not intersect")
                        self.config.pys_print(explain_validity(p_roof))
                        sys.exit(0)

                    # Intersection over minimum area
                    if inter.area / float(min(p_roof_area, gt_roof_area)) >= 0.5:
                        graph.add_edge(p_i, igt_idx)

        return p_roofs, gt_roofs, graph

    def build_graph_2d(self, spatial_index, graph):
        gt_polygons = []

        for i, gt in self.gt_surfaces.iterrows():

            list_features, graph = feedGraph(
                gt, graph, len(gt_polygons), self.config.gt_id, "gt"
            )

            for feature in list_features:
                gt_polygons.append(feature)
                try:
                    bbox_gt = rasterio.features.bounds(feature)
                except TypeError:
                    self.config.pys_print(
                        "Warning: One geometry might be empty, id : "
                        + str(gt[self.config.gt_id])
                    )
                    continue

                spatial_index.insert(id=len(gt_polygons) - 1, coordinates=bbox_gt)

        # Insert the Predictions in the graph
        list_entities = (
            gt_polygons + []
        )  # for now a list of all entities that we will purge after

        for i, p in self.p_surfaces.iterrows():

            list_features, graph = pyscoring.utils.feedGraph(
                p, graph, len(list_entities), self.config.p_id, "p"
            )
            for feature in list_features:
                list_entities.append(feature)

                # Matching this P with every GT possible
                graph = self.matcher(
                    feature, len(list_entities) - 1, gt_polygons, spatial_index, graph
                )

        p_polygons = list_entities[len(gt_polygons) :]

        return p_polygons, gt_polygons, graph

    def matcher(self, polygon, id_polyg_in_graph, list_to_match, geo_index, graph):
        """
        :param extend_matches:
        :param polygon:
        :param id_polyg_in_graph:
        :param list_to_match:
        :param geo_index:
        :param graph:
        :return:
        """
        bbox_p = rasterio.features.bounds(polygon)

        matched = False
        list_possible_matches = []
        list_possible_index = []
        best_I = 0
        id_best_I = -1
        # for each gt_bbox (in the index) intersecting the p_bbox, calculate IoMA
        for number_index in geo_index.intersection(bbox_p):

            possible_match = list_to_match[number_index]
            list_possible_matches.append(possible_match)
            list_possible_index.append(number_index)
            try:
                possible_match_union = union_all(possible_match)
                I = intersection(polygon, possible_match_union)
                if I.area > best_I:
                    best_I = I.area
                    id_best_I = number_index
                ioma = I.area / float(min(polygon.area, possible_match_union.area))
                if ioma >= 0.5:
                    graph.add_edge(id_polyg_in_graph, number_index, weight=ioma)
                    matched = True
            except TopologicalError:
                pass
            except GEOSException as E:
                self.config.pys_print(E)
                self.config.pys_print(
                    "Invalid intersection "
                    + graph.nodes[id_polyg_in_graph]["obj"]
                    + " :"
                    + str(polygon)
                )
        if self.config.extended and not matched and len(list_possible_matches) > 1:
            union_matches = shapely.unary_union(list_possible_matches)
            try:
                I = intersection(polygon, union_matches)
                ioma = I.area / float(min(polygon.area, union_matches.area))
                if ioma >= 0.5 and id_best_I != -1:
                    graph.add_edge(id_polyg_in_graph, id_best_I, weight=ioma)
            except TopologicalError:
                pass
        return graph

    def get_surfaces(self, file_path, label=None, epsg=None):
        if self.config.mode == 2:
            return self.get_surfaces_2d(file_path, label, epsg)
        else:
            return self.get_surfaces_3d(file_path)

    def get_surfaces_3d(self, file_path):
        # Read input file data
        data = cityjson.load(file_path)
        # Extract Builings and BuildingParts
        building_items = data.get_cityobjects(type=["building", "buildingpart"])
        # Extract 2D GroundSurfaces
        groundsurface_polygons_2d, groundsurface_2d_global_bbox = (
            self.extract_CJIO_surfaces(
                building_items, list_type_surface=["groundsurface"], lod=2, z=False
            )
        )
        # Extract 3D RoofSurfaces and OuterFloorSurfaces
        roofsurface_polygons_3d, roofsurface_3d_global_bbox = (
            self.extract_CJIO_surfaces(
                building_items,
                list_type_surface=["outerfloorsurface", "roofsurface"],
                lod=2,
                z=True,
            )
        )

        return roofsurface_polygons_3d, groundsurface_2d_global_bbox

    def get_surfaces_2d(self, file_path, label, epsg):
        # Read input file data
        data = gpd.read_file(file_path)
        # Test the CRS :
        if epsg:
            crs_target = str("EPSG:") + epsg
            crs_target = pyproj.CRS(crs_target)
            proj4 = crs_target.to_proj4()

            crs_origin = str(data.crs)
            crs_origin = pyproj.CRS(crs_origin)

            trans_to_origin = pyproj.Transformer.from_crs(
                crs_target, crs_origin, always_xy=True
            ).transform

        else:
            proj4 = data.crs.to_proj4()
            trans_to_origin = None

        if proj4.split("+proj=")[1].split(" ")[0] == "longlat":
            self.config.pys_print(
                "Error the chosen EPSG (the data if no EPSG chosen) is a geographic system, projected system needed."
            )
            sys.exit(0)

        # Transform if necessary
        if epsg:
            data = data.to_crs(epsg)

        # Extract the selected label
        if label:
            data = select_label(data, label)

        # Get BBOX
        bounds = rasterio.features.bounds(data)
        return [data, bounds, data.crs, trans_to_origin]

    def make_polygons_valid(self, polygons_list):
        valid_polygons = []
        new_polygons_count = 0
        for id, polygon in enumerate(polygons_list):
            # Try make polygon valid
            valid_polygon = make_valid(polygon)
            # If make_valid result is multiple polygons or multiple geometries
            if (
                valid_polygon.geom_type == "MultiPolygon"
                or valid_polygon.geom_type == "GeometryCollection"
            ):
                for p in valid_polygon.geoms:
                    if p.geom_type == "Polygon":
                        valid_polygons.append(p)
                        new_polygons_count += 1
            # else if make_valid result is directly valid and of type polygon
            elif valid_polygon.geom_type == "Polygon":
                valid_polygons.append(valid_polygon)
            # The make_valid result if not of polygon type
            # We don't know how to deal with it
            else:
                self.config.pys_print("\tCould not make polygon valid " + str(id))
                # print("\t"+str(polygon))

        return valid_polygons, new_polygons_count

    def extract_CJIO_surfaces(self, building_items, list_type_surface, lod, z):
        extracted_surfaces = {}
        bounds = None

        # For each building :
        for building_key, building_value in building_items.items():
            polygons = []

            # If building has no geometry, continue
            if len(building_value.geometry) == 0:
                continue

            # Process all geometries
            for geom in building_value.geometry:
                for type in list_type_surface:
                    if len(geom.surfaces) == 0:
                        continue

                    surfaces = geom.get_surfaces(type=type, lod=lod)
                    for surface_key, surface in surfaces.items():
                        boundaries = geom.get_surface_boundaries(surface)
                        for boundary in list(boundaries):
                            polygon = Polygon(boundary[0])
                            if not z:
                                polygon = force_2d(polygon)
                            polygons.append(polygon)
                            if not bounds:
                                bounds = polygon.bounds
                            else:
                                bounds = merge_bounds(
                                    bounds, polygon.bounds, self.config.resolution
                                )

            extracted_surfaces[building_key] = polygons

        return extracted_surfaces, list(bounds)


def select_label(data, label):
    for idx, row in data.iterrows():
        if row["num_class"] != label and row["num_class"] != str(label):
            data = data.drop(index=idx)
    return data


def merge_bounds(first_bound, second_bound, resolution=0.5):
    new_bound = [0, 0, 0, 0]
    new_bound[0] = min(first_bound[0], second_bound[0])
    new_bound[1] = min(first_bound[1], second_bound[1])
    new_bound[2] = max(first_bound[2], second_bound[2])
    new_bound[3] = max(first_bound[3], second_bound[3])

    # if resolution provided then  make the bbox a multiple of the resolution
    if resolution > 0:
        new_bound[0] = floor(new_bound[0])
        new_bound[1] = floor(new_bound[1])
        new_bound[2] = ceil(new_bound[2] / resolution) * resolution
        new_bound[3] = ceil(new_bound[3] / resolution) * resolution

    return tuple(new_bound)


def exclusion(data, bbox, exclusion_strict=False):

    if exclusion_strict:
        border = bbox.boundary
        intersects = data.intersects(border)
        for idx, row in data.iterrows():
            if intersects.get(idx):
                data = data.drop(index=idx)

    outside_elements = data.disjoint(bbox)

    for idx, row in data.iterrows():
        if outside_elements.get(idx):
            data = data.drop(index=idx)

    return data


def multi2poly(json_multi):
    """

    :param json_multi: a mapped MultiPolygon

    :return: a list of Polygon
    """
    list_poly = []

    for coords in json_multi["coordinates"]:
        shell = coords[0]
        rings = []

        if len(coords) > 1:
            rings = [coords[i] for i in range(1, len(coords))]

        new_feature = Polygon(shell=shell, holes=rings)

        list_poly.append(new_feature)

    return list_poly


def feedGraph(feature, graph, i, id, nature):
    json_ft = mapping(feature["geometry"])

    if json_ft["type"] == "MultiPolygon":
        list_polygons = multi2poly(json_ft)
    else:
        list_polygons = [feature["geometry"]]

    for p in range(len(list_polygons)):
        if id:
            id_data = (
                str(feature[id])
                if len(list_polygons) == 1
                else str(feature[id]) + "_" + str(p)
            )
        else:
            id_data = None

        graph.add_node(i + p, nature=nature, obj=id_data)

    return list_polygons, graph


def to_coordinates(feature):
    """Make valid coordinates out of a GeoJSON feature."""
    shp_coordinates = feature["coordinates"]
    coordinates = []

    for shp_ring in shp_coordinates:
        ring = []
        for xy in shp_ring:
            ring.append(list(xy))
        coordinates.append(ring)

    return coordinates
