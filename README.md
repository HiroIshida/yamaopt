# yamaopt
Optimization of sensor placement considering robot forward-kinematics reachability.
The ros wrapper for this package can be found in https://github.com/708yamaguchi/yamaopt_ros.

Given urdf and multiple polygons and observation point, the program determines the optimal sensor placement that maximizes the distance between the observation point and sensor, satisfying the robot's kinematic constraint.

## Installation

If you use python 2.x, following installation is required to install scikit-robot.
```
sudo apt-get install -y libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev
pip install scikit-build
```

Yamaopt can be installed by
```
git clone https://github.com/HiroIshida/yamaopt.git
cd yamaopt
pip install -e .
```

## Usage
Write your own config file defining `urdf_path`, `contorl_joint_names`, `endeffector_link_name`. (see [/config](/config)).

The typical code for yamaopt's optimization is as follows:

```python
from yamaopt.solver import KinematicSolver, SolverConfig

config = SolverConfig.from_config_path(config_path, use_base=use_base, optframe_xyz_from_ef=[0, 0.5, 0.0])
kinsol = KinematicSolver(config)

Note that `polygon` in `example_multi.py` is a 

sol = kinsol.solve_multiple(q_init, polygons, target_obj_pos,
                            d_hover=d_hover,
                            joint_limit_margin=joint_limit_margin)
angle_vector_solution = sol.x
```

- `optframe_xyz_from_ef` determines the sensor's sweet spot coordinate w.r.t the end-effector coordinate. 

- If `use_base` is `True`, base's 3dof (2d position and yaw angle) is also considered.

- `q_init`: initial joint angles for each control joint. The length of this vector must match that of `contorl_joint_names` defined in the config file.

- `polygons`: list of polygons. Each polygon is represented by numpy matrix. If there are four vertex, it is a 4 x 3 matrix. Vertexes must be clockwise or counter-clockwise order.

- `d_hover`: The distance in meter between the solution end-effector position from the polygon surface.

- `joint_limit_margin`: joint limit margin from the joint bounds in the urdf.

## Demo
```
cd examples
python3 example_multi.py --visualize --use_base -robot pr2
```
<img src="https://user-images.githubusercontent.com/38597814/155905002-4834a833-5220-40a9-9eef-3cf03f59a06e.png" width="60%" />

- blue sphere: observation point: `target_pos`

- yellow sphere: sensor's 'sweet spot'

- red and green polygons: input polygons

- green polyon: polygon that sensor will be attached to
