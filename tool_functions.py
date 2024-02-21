"""

Get the coordinates, hash and estimated cost of a new node

author: Tianlong Zhang (SWJTU)

see https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12990 for transition curve calculation explanation

"""

from math import sqrt, cos, sin, tan, atan, pi


def pi_2_pi(angle):
    """Turn the angle between -pi and pi"""
    return (angle + pi) % (2 * pi) - pi


def calc_index(node, c):
    """Calculate the hash value of the node"""
    ind = (
        (node.gradient_z_ind - c.min_gradient_z) * c.theta_xyw * c.zw * c.yw * c.xw
        + (node.theta_xy_ind - c.min_theta_xy) * c.zw * c.yw * c.xw
        + (node.zind - c.minz) * c.xw * c.yw
        + (node.yind - c.miny) * c.xw
        + (node.xind - c.minx)
    )

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def calc_cost_new(current, heuristic):
    """Calculate the total cost of f(), f(x) = g(x) + h(x)"""
    cx, cy, cz = current.xlist[-1], current.ylist[-1], current.zlist[-1]
    return current.cost + 0.95 * heuristic[round(cx), round(cy)]


def z_y_move(x, y, z, theta_xy, gradient_z, distance, radi):
    """Straight line or circle exploration"""
    if radi is not None:
        r = radi
        theta = pi_2_pi(distance / r)
        tmp_x = r * sin(theta)
        tmp_y = r - r * cos(theta)
        new_r = sqrt(tmp_x**2 + tmp_y**2)
        theta0 = theta_xy + atan(tmp_y / tmp_x)
        x += new_r * cos(theta0)
        y += new_r * sin(theta0)
        z += distance * gradient_z
        theta_xy += theta
    else:
        x += distance * cos(theta_xy)
        y += distance * sin(theta_xy)
        z += distance * gradient_z

    return x, y, z, pi_2_pi(theta_xy), gradient_z


def spr_move1(x, y, z, theta_xy, gradient_z, distance, len_s, radi):
    """transition curve from line to curve"""
    ls, l = len_s, distance
    r = radi
    # -(79*l**12)/(2043241200*r**6*ls**6))  # 弦长
    c = l * (
        1 - (l**4) / (90 * r**2 * ls**2) + (l**8) / (22680 * r**4 * ls**4)
    )
    # -(23*l**12)/(1915538625*r**6*ls**6))  # 弦切角
    theta = (
        (l**2)
        / (r * ls)
        * (
            1 / 6
            - (l**4) / (2835 * r**2 * ls**2)
            - (l**8) / (467775 * r**4 * ls**4)
        )
    )
    beta = (l**2) / (2 * r * ls)  # 切线角
    theta0 = theta_xy + theta
    x += c * cos(theta0)
    y += c * sin(theta0)
    z += distance * gradient_z
    theta_xy += beta
    return x, y, z, pi_2_pi(theta_xy), gradient_z


def spr_move2(x, y, z, theta_xy, gradient_z, distance, len_s, radi=None):
    """transition curve from curve to line"""
    ls, l = len_s, len_s
    r = radi
    # -(79*l**6)/(2043241200*r**6))
    c = l * (1 - (l**2) / (90 * r**2) + (l**4) / (22680 * r**4))
    # -(23*l**6)/(1915538625*r**6))
    theta = (
        (l) / (r) * (1 / 6 - (l**2) / (2835 * r**2) - (l**4) / (467775 * r**4))
    )
    beta = (l) / (2 * r)  # 切线角
    theta0 = theta_xy + (beta - theta)
    x_final = x + c * cos(theta0)
    y_final = y + c * sin(theta0)
    theta_xy_final = pi_2_pi(theta_xy + beta)

    ls, l = len_s, len_s - distance
    # -(79*l**12)/(2043241200*r**6*ls**6))  # 弦长
    c = l * (
        1 - (l**4) / (90 * r**2 * ls**2) + (l**8) / (22680 * r**4 * ls**4)
    )
    # -(23*l**12)/(1915538625*r**6*ls**6))  # 弦切角
    theta = (
        (l**2)
        / (r * ls)
        * (
            1 / 6
            - (l**4) / (2835 * r**2 * ls**2)
            - (l**8) / (467775 * r**4 * ls**4)
        )
    )
    beta = (l**2) / (2 * r * ls)  # 切线角
    theta0 = theta_xy_final + pi - theta
    x = x_final + c * cos(theta0)
    y = y_final + c * sin(theta0)
    z = z + distance * gradient_z
    theta_xy = theta_xy_final - beta

    return x, y, z, pi_2_pi(theta_xy), gradient_z
