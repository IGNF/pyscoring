import os
import sys

import numpy as np
from shapely.geometry import Point
from math import floor, degrees, atan2, sqrt, ceil
import rasterio.features


class ZMetric:
    r""" """

    def __init__(
        self,
        trim_invalid_geometry=False,
        autocorrect_invalid_geometry=False,
        epsg=None,
        start_index=-1000000,
        no_data=-9999,
        save_raster=None,
    ):

        self.trim_invalid_geometry = trim_invalid_geometry
        self.autocorrect_invalid_geometry = autocorrect_invalid_geometry

        self.start_index = start_index
        self.no_data = no_data
        self.save_raster_pth = save_raster
        self.crs = str("EPSG:") + epsg
        self._init_values()

    def _init_values(self):

        # Stats pre-metrics
        self.nb_pix = 0
        self.nb_pix_only_gt = 0
        self.nb_pix_only_p = 0
        self.nb_pix_ground_gt = 0
        self.nb_pix_ground_p = 0

        self.diff_z = 0
        self.diff_z_sq = 0
        self.nb_pix_masked = 0  # errors

        # Metrics
        self.under_rc = 0
        self.over_rc = 0
        self.sm_err = 0
        self.sm_err_sq = 0

    def update(self, predictions, ground_truths, boundingbox=None, resolution=0.5):
        """
        Accumulates metrics for new detections and ground truths

        :param predictions: list of shapely.Polygon, the new detections to evaluate
        :param ground_truths: list of shapely.Polygon, the new ground truths to which we compare the detections
        :param boundingbox: list of coordinates defining the working zone to take into account in the metrics
        :param resolution: float, resolution to which we want to consider to surfacic evaluation of the data,
                            e.g. the resolution of the data source
        """

        # Compute width and height in pixels
        if boundingbox is None:
            # Select the union of the two BBOX
            bboxp = None
            bboxg = None

            for p in predictions:
                if bboxp is None:
                    bboxp = list(p.bounds)
                else:
                    bboxp[0] = min(bboxp[0], list(p.bounds)[0])
                    bboxp[1] = min(bboxp[1], list(p.bounds)[1])
                    bboxp[2] = max(bboxp[2], list(p.bounds)[2])
                    bboxp[3] = max(bboxp[3], list(p.bounds)[3])
            bboxp = [
                bboxp[0] - resolution,
                bboxp[1] - resolution,
                bboxp[2] + resolution,
                bboxp[3] + resolution,
            ]

            for g in ground_truths:
                if bboxg is None:
                    bboxg = list(g.bounds)
                else:
                    bboxg[0] = min(bboxg[0], list(g.bounds)[0])
                    bboxg[1] = min(bboxg[1], list(g.bounds)[1])
                    bboxg[2] = max(bboxg[2], list(g.bounds)[2])
                    bboxg[3] = max(bboxg[3], list(g.bounds)[3])
            bboxg = [
                bboxg[0] - resolution,
                bboxg[1] - resolution,
                bboxg[2] + resolution,
                bboxg[3] + resolution,
            ]

            boundingbox = [0, 0, 0, 0]
            boundingbox[0] = floor(min(bboxp[0], bboxg[0]))
            boundingbox[1] = floor(min(bboxp[1], bboxg[1]))
            boundingbox[2] = ceil(max(bboxp[2], bboxg[2]) / resolution) * resolution
            boundingbox[3] = ceil(max(bboxp[3], bboxg[3]) / resolution) * resolution

        width = int((boundingbox[2] - boundingbox[0]) / resolution)
        height = int((boundingbox[3] - boundingbox[1]) / resolution)

        output_shape = (height, width)
        self.nb_pix += width * height
        trsf = rasterio.transform.from_bounds(*boundingbox, width, height)

        # Initialising rasters with polygons indices, no_data = 0
        # ID rasters

        p_ids_raster = rasterio.features.rasterize(
            predictions, out_shape=output_shape, transform=trsf, dtype="float32"
        )
        gt_ids_raster = rasterio.features.rasterize(
            ground_truths, out_shape=output_shape, transform=trsf, dtype="float32"
        )

        if self.save_raster_pth:
            self.save_raster(p_ids_raster, "p_roofs_id.tif", width, height, trsf)
            self.save_raster(gt_ids_raster, "gt_roofs_id.tif", width, height, trsf)

        # Ground masks
        ground_p_mask = np.ma.masked_values(p_ids_raster, 0).mask
        ground_gt_mask = np.ma.masked_values(gt_ids_raster, 0).mask
        ground_mask = np.logical_or(ground_p_mask, ground_gt_mask)

        if self.save_raster_pth:
            self.save_raster(ground_p_mask, "p_mask.tif", width, height, trsf)
            self.save_raster(ground_gt_mask, "gt_mask.tif", width, height, trsf)
            self.save_raster(ground_mask, "ground_mask.tif", width, height, trsf)

        # Find pixels where a polygon overlaps ground
        superposition = np.logical_xor(ground_gt_mask, ground_p_mask)

        # Find specifically where groundtruths pixels're overlapping ground, and derive from it the under reconstruction
        superposition_mask_gt_alone = np.logical_and(
            superposition, np.logical_not(ground_gt_mask)
        )
        self.nb_pix_ground_gt += np.sum(ground_gt_mask)
        self.nb_pix_only_gt += float(np.ma.sum(superposition_mask_gt_alone))

        # Find specifically where detections pixels're overlapping ground, and derive from it the over reconstruction
        superposition_mask_p_alone = np.logical_and(
            superposition, np.logical_not(ground_p_mask)
        )
        self.nb_pix_ground_p += np.sum(ground_p_mask)
        self.nb_pix_only_p += float(np.ma.sum(superposition_mask_p_alone))

        if self.save_raster_pth:
            self.save_raster(
                superposition_mask_p_alone,
                "over_reconstruction_mask.tif",
                width,
                height,
                trsf,
            )
            self.save_raster(
                superposition_mask_gt_alone,
                "under_reconstruction_mask.tif",
                width,
                height,
                trsf,
            )
            self.save_raster(superposition, "superposition.tif", width, height, trsf)

        # Compute the Z coordinates in the rasters
        p_z_raster, cnt_roofs_too_small = self.zmask(
            p_ids_raster, boundingbox, predictions, resolution, self.no_data
        )
        mask_errors_p = np.ma.masked_values(p_z_raster, self.no_data)

        gt_z_raster, cnt_roofs_too_small = self.zmask(
            gt_ids_raster, boundingbox, ground_truths, resolution, self.no_data
        )
        mask_errors_gt = np.ma.masked_values(gt_z_raster, self.no_data)

        if self.save_raster_pth:
            self.save_raster(p_z_raster, "p_roofs_z.tif", width, height, trsf)
            self.save_raster(gt_z_raster, "gt_roofs_z.tif", width, height, trsf)

        # Propagation the error masks
        # Will be 'nodata' where there is :
        # - ground
        # - errors (polygons too small)
        # - no superposition (e.g. : a pixel  noted roof in detection and ground in gt, or vice versa

        fusion_errors_masks = np.logical_or(mask_errors_gt.mask, mask_errors_p.mask)
        fusion_error_superposition = np.logical_or(fusion_errors_masks, superposition)
        fusion_all_masks = np.logical_or(ground_mask, fusion_error_superposition)
        mask_errors_gt.mask = fusion_all_masks
        mask_errors_p.mask = fusion_all_masks

        # Calculation of the Z difference between the two masked rasters
        mask_diff = np.ma.subtract(mask_errors_gt, mask_errors_p, dtype=np.float32)
        
        self.nb_pix_masked += np.sum(fusion_all_masks)

        if self.save_raster_pth:
            self.save_raster(mask_diff, "diff_z.tif", width, height, trsf, self.no_data)
            self.save_raster(mask_diff.mask, "mask_errors.tif", width, height, trsf)

        self.diff_z += np.ma.sum(mask_diff)
        self.diff_z_sq += np.ma.sum(np.ma.power(mask_diff, 2))

    def compute(self):
        """

        :return:
        """

        self.under_rc = self.nb_pix_only_gt / (self.nb_pix - self.nb_pix_ground_gt)
        self.over_rc = self.nb_pix_only_p / (self.nb_pix - self.nb_pix_ground_p)

        self.sum_err = self.diff_z / (self.nb_pix - self.nb_pix_masked)
        self.sum_err_sq = self.diff_z_sq / (self.nb_pix - self.nb_pix_masked)

        return self.under_rc, self.over_rc, self.sum_err, sqrt(self.sum_err_sq)

    def zmask(self, z_mask, bbox, roof_and_ids, resolution, nodata, margin=1):
        """
        :param z_mask:
        :param bbox:
        :param roof_and_ids:
        :param resolution:
        :param nodata:
        :param margin:
        :return:
        """
        xmin, ymin, xmax, ymax = bbox

        roof_area_too_small = 0

        for roof_and_id in roof_and_ids:
            roof, roof_id = roof_and_id

            # bbox of the working zone and a wee margin
            xn, yn = max(roof.bounds[0] - margin, xmin), max(
                roof.bounds[1] - margin, ymin
            )
            xm, ym = min(roof.bounds[2] + margin, xmax), min(
                roof.bounds[3] + margin, ymax
            )

            # Pixel coordinates
            _in = floor((ymax - yn) / resolution)
            _jn = floor((xn - xmin) / resolution)

            # Last i;j of the working zone
            _im = floor((ymax - ym) / resolution)
            _jm = floor((xm - xmin) / resolution)

            # If roof polygon area is close to zero no need to go further
            if roof.area < resolution * resolution:
                for i in range(_im, _in):
                    for j in range(_jn, _jm):
                        if z_mask[i][j] == roof_id:
                            roof_area_too_small += 1
                            z_mask[i][j] = nodata
                continue

            # Keep only polygon corners
            min_angle = 175
            max_angle = 185
            delta = 0.5

            try:
                if len(roof.exterior.coords) == 4:
                    roof_corners = roof.exterior.coords[:-1]
                else:
                    roof_corners = self.getThreeGoodCorners(
                        roof, min_angle, max_angle, 0.01
                    )
            except Exception as e:
                print(e)
                print("Polygon : " + str(roof))
                continue

            while len(roof_corners) != 3 and max_angle > 180.5:
                min_angle += delta
                max_angle -= delta
                roof_corners = self.getThreeGoodCorners(
                    roof, min_angle, max_angle, 0.01
                )

            if len(roof_corners) != 3:
                print("Could not find three good corners in this polygon")
                print("polygon  area " + str(roof.area))
                print(roof)
                print(
                    "Mask out this polygon if ever the center of a pixel of the raster is in it"
                )
                num_masked = 0

                for i in range(_im, _in):
                    for j in range(_jn, _jm):
                        if z_mask[i][j] == roof_id:
                            roof_area_too_small += 1
                            z_mask[i][j] = nodata
                            num_masked += 1
                print(str(num_masked) + " pixels masked for this polygon")
                continue

            # Get the plan equation
            a, b, c, d = self.get_plan_equation(roof_corners)

            # Stop everything if wrong equation
            if c == 0:
                print("Could not get a plan equation")
                print(roof_corners)
                sys.exit(1)

            # Keep points to visualize in case of error
            points_in_poly = []

            # For every pixel in the bbox surrounding the roof
            for i in range(_im, _in):
                for j in range(_jn, _jm):
                    center_pix = [
                        xmin + (j + 1 / 2) * resolution,
                        ymax - (i + 1 / 2) * resolution,
                    ]
                    points_in_poly.append(Point(center_pix))
                    # Check if the pixel's center is inside the roof polygon

                    if z_mask[i][j] == roof_id:
                        z = self.get_z(a, b, c, d, center_pix)
                        # If altitude is negative, could be an error, let's test it
                        if z < 0:
                            test_z_neg = False
                            # Check if one the polygon's points has a negative altitude
                            for poly_pt in roof.exterior.coords[:-1]:
                                if poly_pt[2] < 0:
                                    test_z_neg = True
                                    break

                            # If so everything is ok we can continue
                            if test_z_neg:
                                pass
                            # If not, try and simplify the polygon to get three good corners for pan equation
                            # TODO : maybe find a better strategy
                            else:
                                roof_corners = self.getThreeGoodCorners(
                                    roof.simplify(0.05), 175, 185, 0.01
                                )
                                a, b, c, d = self.get_plan_equation(roof_corners)
                                z = self.get_z(a, b, c, d, center_pix)
                                # If z is still negative, raise an error and exit
                                if z < 0:
                                    print("Could not deal w this polygon. Exiting.")
                                    corner_pts = [
                                        Point(corner) for corner in roof_corners
                                    ]
                                    print(corner_pts)
                                    print(Point(center_pix))
                                    print(z)
                                    print(a, b, c, d)
                                    sys.exit(1)
                                else:
                                    pass
                        z_mask[i][j] = z

        return z_mask, roof_area_too_small

    def get_plan_equation(self, corners):
        """
        Get the equation for a plan constitued of the corners
        :param corners: list of 3 points in the plan
        :return: the paramets a, b, c, d of the equation of the corresponding plan
        """
        # Get coordinates
        A, B, C = corners

        # Get points
        p1 = np.array(A)
        p2 = np.array(B)
        p3 = np.array(C)

        # Get vectors
        v1 = p3 - p1
        v2 = p2 - p1

        # Get the equation plan
        cp = np.cross(v1, v2)
        (
            a,
            b,
            c,
        ) = cp
        d = np.dot(cp, p3)

        return a, b, c, d

    def getThreeGoodCorners(self, polygon, min_angle, max_angle, min_distance):
        """
        Select 3 corners in a polygon corresponding to characteristics : not to small nor big and with enough distance between them

        :param polygon: Shapely polygon
        :param min_angle: int or float, the minimum angle
        :param max_angle: int or float, the maximum angle
        :param min_distance: int or float, the minimum distance between two selected corners
        :return: list of 3 corners of the polygon
        """
        corners = []
        exterior_points = polygon.exterior.coords[:-1]

        for i in range(len(exterior_points)):

            a = self.getElt(exterior_points, i - 1)
            b = self.getElt(exterior_points, i)
            c = self.getElt(exterior_points, i + 1)

            angle = self.getAngle(a, b, c)

            if angle > min_angle and angle < max_angle:
                continue

            if b not in corners:
                corners.append(b)

        # Remove the points too close one to another
        result = []
        result.append(corners[0])
        for i in range(1, len(corners)):
            dst = Point(corners[i]).distance(Point(corners[i - 1]))
            if dst > min_distance:
                result.append(corners[i])
            if len(result) == 3:
                break

        return result

    def getElt(self, list, index):
        """
        Get element in list to handle a sliding window
        :param list: list
        :param index: index to select
        :return: the element at the index
        """

        if index < 0:
            index = len(list) + index
        elif index > len(list) - 1:
            index = index - len(list)

        return list[index]

    def getAngle(self, a, b, c):
        """
        Return angle between three points
        :param a: point A
        :param b: point B
        :param c: point C
        :return: the angle in degrees
        """
        angle = degrees(
            atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0])
        )
        return angle + 360 if angle < 0 else angle

    def reset(self):
        self._init_values()

    def get_z(self, a, b, c, d, P):
        """
        From 4 equation parameters and a Point(X,Y) get the Z corresponding to the Point
        :param a: a parameter
        :param b: b parameter
        :param c: c parameter
        :param d: d parameter
        :param P: Point(X, Y)
        :return: Z
        """

        x = P[0]
        y = P[1]

        try:
            z = (d - a * x - b * y) / c
        except Exception as e:
            z = 0
        return z

    def save_raster(self, raster, name, width, height, trsf, nodata=None):
        with rasterio.open(
            os.path.join(self.save_raster_pth, name),
            "w",
            driver="GTiff",
            dtype=rasterio.float32,
            count=1,
            width=width,
            height=height,
            crs=self.crs,
            transform=trsf,
            nodata=nodata,
        ) as dst:
            dst.write(raster, indexes=1)
