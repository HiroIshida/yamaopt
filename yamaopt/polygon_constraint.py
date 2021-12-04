import attr
import numpy as np
from geometry_msgs.msg import PolygonStamped

def polygonstamped_to_nparray(polygon: PolygonStamped):
    return np.array([[pt.x, pt.y, pt.z] for pt in polygon.polygon.points])

@attr.s # like a dataclass in python3
class LinearEqConst(object):
    # A * x - b = 0
    A = attr.ib()
    b = attr.ib()
    def __call__(self, x): return self.A.dot(x) - self.b 
    def is_satisfying(self, x): return np.all(np.abs(self.__call__(x)) < 1e-7)

@attr.s
class LinearIneqConst(object):
    # A * x - b >= 0
    A = attr.ib()
    b = attr.ib()
    def __call__(self, x): return self.A.dot(x) - self.b 
    def is_satisfying(self, x): return np.all(self.__call__(x) > -1e-7)

def check_convexity_and_maybe_ammend(np_polygon):
    # TODO(HiroIshida) PR to jsk_pcl_ros
    # to circumvent jsk_pcl_ros's bag. Ad-hoc fix to convert non-convex to convex
    points = np_polygon
    points_auged = np.vstack([points, points[0], points[1]])

    dotpro_list = []
    for i in range(len(points)):
        vec1 = points_auged[i+1] - points_auged[i]
        vec2 = points_auged[i+2] - points_auged[i+1]
        dotpro_list.append(vec1.dot(vec2))
    dotpro_list = np.array(dotpro_list)

    positive_idxes = np.where(dotpro_list >= 0)[0]
    negative_idxes = np.where(dotpro_list < 0)[0]
    if len(positive_idxes)==0 or len(negative_idxes) == 0:
        return np_polygon

    if min(len(positive_idxes), len(negative_idxes)) > 1:
        raise RuntimeError("not convex. and cannot fix by the adhoc method")
    return np_polygon

def is_convex(np_polygon):
    points = np_polygon
    points_auged = np.vstack([points, points[0], points[1]])

    crosspro_list = []
    for i in range(len(points)):
        vec1 = points_auged[i+1] - points_auged[i]
        vec2 = points_auged[i+2] - points_auged[i+1]
        crosspro_list.append(np.cross(vec1, vec2))
    sign_list = np.sign([crosspro_list[0].dot(e) for e in crosspro_list])
    return np.all(sign_list > 0) or np.all(sign_list < 0)


def polygon_to_constraint(np_polygon):
    assert is_convex(np_polygon)

    normalize = lambda vec: vec/np.linalg.norm(vec)

    points = np_polygon
    points_auged = np.vstack([points, points[0]])

    n_vec = normalize(np.cross(points[2] - points[1], points[1] - points[0]))
    # construct equality constraint
    p_whatever = points[0]
    A_eq = np.array([n_vec])
    b_eq = np.array([n_vec.dot(p_whatever)])
    lineq = LinearEqConst(A_eq, b_eq)

    # construct inequality constraint
    A_ineq_local_list = []
    b_ineq_local_list = []
    for i in range(len(points_auged) - 1):
        p_here = points_auged[i]
        p_next = points_auged[i+1]
        vec = p_next - p_here
        n_vec_local = -normalize(np.cross(n_vec, vec))

        # let q be a query point. Then ineq const is (q - p_here)^T \dot n_vec_local > 0
        A_local = np.array(n_vec_local)
        b_local = np.dot(p_here, n_vec_local)
        A_ineq_local_list.append(A_local)
        b_ineq_local_list.append(b_local)
    A_ineq = np.vstack(A_ineq_local_list)
    b_ineq = np.array(b_ineq_local_list)
    linineq = LinearIneqConst(A_ineq, b_ineq)
    return linineq, lineq

if __name__=='__main__':
    import pickle
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with open('./accum_polygons.pickle', 'rb') as f:
        polygons = pickle.load(f)
    polygons = [polygonstamped_to_nparray(e) for e in polygons]

    polygon = np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    eq, ineq = polygon_to_constraint(polygon)

    for polygon in polygons:
        try:
            eq, ineq = polygon_to_constraint(polygon)
        except Exception as e:
            pass