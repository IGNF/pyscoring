Config file
================
.. code-block:: console

    [MODE]
    MODE = 3

    [PATHS]
    GT_PATH = ./data/gt/gt.json
    P_PATH =  ./data/p/p.json
    OUTPUT_PATH = ./output/output.csv

    [DATA]
    GT_CLASS_COL =
    GT_CLASS_VAL =
    P_CLASS_COL =
    P_CLASS_VAL =
    GT_ID_COL =
    P_ID_COL =
    RES = 0.2
    WBBOX_PATH =

    [PARAMETERS]
    STRICT_BBOX = 1
    EPSG = 2154
    EXTEND_MATCHES = 1
    WORKERS = 5
    CRT = ioma
    TSD = 0.5

    [OUTPUT]
    LOG = 1
    LOCAL = 1
    GLOBAL = 1

MODE
---------
PyScoring can work with 3D and to 2D data.
You can choose here if you want to use it for 3D data MODE = 3 and for 2D data MODE = 2.

GT_PATH
--------
The path where is located the ground truth (GT) data.

P_PATH
-------
The path where is located the prediction (P) data, the data you want to evaluate by comparing it to the ground truth.

OUTPUT_PATH
--------------
The path where PyScoring should save its output files.

GT_CLASS_COL
----------------
If your GT data is multi-class ans only one type of class matters you can specify here where the class information is stored in the GT data.
Let it empty if not.

GT_CLASS_VAL
--------------
If your GT data is multi-class ans only one type of class matters you can specify here on which class focus.
Let it empty if not.

P_CLASS_COL
----------------
If your P data is multi-class ans only one type of class matters you can specify here where the class information is stored in the P data.
Let it empty if not.

P_CLASS_VAL
--------------
If your P data is multi-class ans only one type of class matters you can specify here on which class focus.
Let it empty if not.


GT_ID_COL
------------
If your GT data has an unique ID and you are interested in keeping it, you can specify here where the ID is stored.
Let it empty if not.

P_ID_COL
------------
If your GT data has an unique ID and you are interested in keeping it, you can specify here where the ID is stored.
Let it empty if not.

RES
---------
The resolution of the data, in meters. Set by default to 0.5m.

WBBOX_PATH
-------------
If you have a polygon shapefile defining the working bounding box you want to work on, you can specify the path of this shapefile here.
By default the working bbox will be created by merging P's one and GT's one.

STRICT_BBOX = 1
----------------------
If PyScoring should exclude of its calculation the polygons crossing the defined working bbox or not, put 1. By default 0.

EPSG
--------
The projected coordinate system to use. If not specified will try to use the GT's one. If not a projected system, will crash out.


EXTEND_MATCHS
----------------
If PyScoring should try to merge the intersections of polygons to find new matches, put 1. By default 0.


WORKERS
----------
How many workers available for the parallelization. By default 1.


CRT
---------
Which criteria should PyScoring use to match the polygons of the different datasets. Only 'ioma' (Intersection Over Minimum Area) and 'iou' (Intersection Over Union) available.
By default 'ioma'.

TSD
----------
Defining the threshold used by the criteria chosen juste before. By default 0.5.

LOG
-----
If you want to have the PyScoring logs in the output, put 1.

LOCAL
-------
If you want to have the PyScoring metrics locally for every single association of polygons in the output, put 1.

GLOBAL
--------
If you want to have the PyScoring global metrics, for the whole wbbox defined, in the output, put 1.
