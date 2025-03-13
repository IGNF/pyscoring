# PyScoring

PyScoring is a tool to evaluate the production of vector data by comparing it to an other dataset considered as the ground truth. PyScoring produces metrics qualifying the comparison between the two datasets. 
It works with both with 2D (only in .shp) and 3D data (only respecting the [CityJSON specifications](https://www.cityjson.org/)).

## Installation

After cloning source, install dependencies and the package

```
virtualenv -p 'python3.10' .ve
source .ve/bin/activate
pip install .
```

## Example
And then to launch the example (possible with both 2D or 3D example) : 
```
cd example/2D
python3 example.py
```
Then you can modify the configuration file in the same directory than the _example.py_ you launched, to try with your own parameters and data. 

For any question or request don't hesitate to open an issue. 
