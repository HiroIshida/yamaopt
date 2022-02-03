import functools
import numpy as np
import operator
import hashlib

def array_cache(f):
    # Because np.array is not hashbale, custom cache wrapper must be defined...
    algo = lambda x: hashlib.md5(x).digest() # TODO candidate: default hash() or xxhash.xxh64()
    cache = {}
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        # NOTE __str__() is too big for nparray
        tobyte = lambda val: val.tostring() if type(val) is np.ndarray else val.__str__().encode()
        strenc = functools.reduce(operator.add, map(tobyte, args))
        if kwargs:
            strenc += functools.reduce(operator.add, map(tobyte, list(kwargs.values())))
        hashval = algo(strenc)
        if not hashval in cache:
            cache[hashval] = f(*args, **kwargs)
        return cache[hashval]
    return wrapped

def scipinize(fun):
    """Scipinize a function returning both f and jac

    For the detail this issue may help:
    https://github.com/scipy/scipy/issues/12692

    Parameters
    ----------
    fun: function
        function maps numpy.ndarray(n_dim,) to tuple[numpy.ndarray(m_dim,),
        numpy.ndarray(m_dim, n_dim)], where the returned tuples is
        composed of function value(vector) and the corresponding jacobian.
    Returns
    -------
    fun_scipinized : function
        function maps numpy.ndarray(n_dim,) to a value numpy.ndarray(m_dim,).
    fun_scipinized_jac : function
        function maps numpy.ndarray(n_dim,) to
        jacobian numpy.ndarray(m_dim, n_dim).
    """

    closure_member = {'jac_cache': None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac

def matrix2quaternion(m):
    """Returns quaternion of given rotation matrix.

    Parameters
    ----------
    m : list or numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    quaternion : numpy.ndarray
        quaternion [w, x, y, z] order

    Examples
    --------
    >>> import numpy
    >>> from skrobot.coordinates.math import matrix2quaternion
    >>> matrix2quaternion(np.eye(3))
    array([1., 0., 0., 0.])
    """
    m = np.array(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def quaternion2rpy(q):
    """Returns Roll-pitch-yaw angles of a given quaternion.

    Parameters
    ----------
    q : numpy.ndarray or list
        Quaternion in [w x y z] format.

    Returns
    -------
    rpy : numpy.ndarray
        Array of yaw-pitch-roll angles, in radian.

    Examples
    --------
    >>> from skrobot.coordinates.math import quaternion2rpy
    >>> quaternion2rpy([1, 0, 0, 0])
    (array([ 0., -0.,  0.]), array([3.14159265, 3.14159265, 3.14159265]))
    >>> quaternion2rpy([0, 1, 0, 0])
    (array([ 0.        , -0.        ,  3.14159265]),
     array([3.14159265, 3.14159265, 0.        ]))
    """
    roll = atan2(
        2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] ** 2 - q[2] ** 2 - q[1] ** 2 + q[0] ** 2)
    pitch = -asin(
        2 * q[1] * q[3] - 2 * q[0] * q[2])
    yaw = atan2(
        2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] ** 2 + q[0] ** 2 - q[3] ** 2 - q[2] ** 2)
    rpy = np.array([yaw, pitch, roll])
    return rpy, np.pi - rpy
