# yamaopt

Optimizing the position where the robot attaches the sensor.

## Install

For Python2.x, the following apt install is required to install scikit-robot:

```
sudo apt-get install -y libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev
```

If you don't have `GLIBCXX_3.4.26`, install following packages: (for tinyik)
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-9
sudo apt install libstdc++6
```

Then, install yamaopt

```
git clone https://github.com/HiroIshida/yamaopt.git
cd yamaopt
pip install -e .
```

## Example
```
cd examples
python example_multi.py --visualize --use_base -robot pr2
```

- The blue sphere is the point you want to observe. (`target_pos`).
- The yellow sphere is the sensor's 'sweet spot'.
- The red polygons define the constraints.
- IK is calculated so that the end-effector is perpendicular to the polygon.
- If we shift the position of the blue sphere, we can see that the IK solution also changes (see below).
- Note that `polygon` in `example.py` is a numpy matrix of polygon vertices in order (clockwise or counter-clockwise). If there are four vertex, it is a 4 x 3 matrix.


<img src="https://user-images.githubusercontent.com/38597814/144719358-c811834e-d467-4523-bee0-bfeee1910923.png" width="200px">
<img src="https://user-images.githubusercontent.com/38597814/144719361-8ba3576c-1466-42b7-963d-d719044da235.png" width="200px">
<img src="https://user-images.githubusercontent.com/38597814/144719364-b2fcb361-74f3-4ba9-b08e-fe079dd9cabe.png" width="200px">
