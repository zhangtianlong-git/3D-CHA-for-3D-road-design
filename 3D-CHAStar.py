import math
import tool_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import heapq
import copy
from get_cross_section_info import get_cross_fill_cut, get_multi_layer_cost

"""constraints and defining parameters"""
MAX_GRADIENT = 0.05  # Maximum slope value
MIN_GRADIENT = 0.005  # Minimum slope
GRADIENT_Z_RESOLUTION = 0.005  # Slope resolution
N_STEER = 10  # Maximum number of horizontal exploring directions
LEN_SPIRAL = 80  # Transition curve length
EXPLORE_RES = 40  # Linear and circular curve exploration Steps
MIN_LEN_SLOPE = 225  # Minimum slope length
MIN_R = 250  # Minimum curve radius
MIN_LEN_CURV = 100  # Minimum curve length
MIN_LEN_TAN = 222  # Shortest straight line length
THETA_XY_RESOLUTION = np.deg2rad(1)  # Horizontal Angle Resolution
K_MIN = 10000  # Vertical curve radius
MAX_LEN_TAN = 1321  # Maximum linear length
"""----------------"""

# Horizontal angle compensation value to avoid negative values
THETA_XY_IND = round(math.pi / THETA_XY_RESOLUTION)
# Vertical slope compensation values to avoid negative values
GRADIENT_Z_IND = round(MAX_GRADIENT / GRADIENT_Z_RESOLUTION)
# Number of line element segments with minimum curve length
MIN_SEG = MIN_LEN_CURV / EXPLORE_RES
# Number of line element segments with minimum line length
MIN_SEG_T = MIN_LEN_TAN / EXPLORE_RES
# Number of segments corresponding to minimum slope length
MIN_SEG_S = MIN_LEN_SLOPE / EXPLORE_RES
# Maximum Turning Angle, which is mainly used to determine the angular resolution
MAX_ANGLE_CHANGE = EXPLORE_RES / MIN_R
# All Turning Angles
ANGLES = list(np.linspace(-MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE, N_STEER))
# All radii
RADIUS = EXPLORE_RES / np.linspace(-MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE, N_STEER)
RADIUS = list(((RADIUS / 10).round(0)) * 10)
# All slope values
GRADIENTS = np.linspace(
    -MAX_GRADIENT, MAX_GRADIENT, int(2 * MAX_GRADIENT / GRADIENT_Z_RESOLUTION + 1)
)
GRADIENTS = list(GRADIENTS[abs(GRADIENTS) >= MIN_GRADIENT - 0.00001])

# Topographic parameters and other very large matrices are first saved as txt,
# and then read, to improve the efficiency of calculations
terrain = np.loadtxt("terrain.txt") * 1000
terrain1 = terrain - 3
heuristic = np.loadtxt("heuristic.txt").T


class Node:
    def __init__(
        self,
        xind,
        yind,
        zind,
        theta_xy_ind,
        gradient_z_ind,
        xlist,
        ylist,
        zlist,
        theta_xy_list,
        gradient_z_list=[0],
        pind=None,
        cost=0,
        line_type=None,
        radius=None,
        seqZ=0,
        seqY=0,
        seqS=0,
        lens=0,
        start=False,
        seqSS=0,
        pre_ver_cur_len=0,
        fill_cut_volunm=[],
        loc_and_len_cost=[],
        cross_section_cost=[],
    ):
        self.xind = xind  # Integer part of the x-coordinate
        self.yind = yind
        self.zind = zind
        self.theta_xy_ind = theta_xy_ind
        self.gradient_z_ind = gradient_z_ind 
        self.xlist = xlist  # x-coordinate，it is a list
        self.ylist = ylist
        self.zlist = zlist
        self.theta_xy_list = theta_xy_list
        self.gradient_z_list = gradient_z_list
        self.pind = pind  # Pointer to the last explored node (parent)
        self.cost = cost  # Total cost
        self.radius = radius 
        self.lens = lens  # Total length from the current node to the start point
        self.line_type = line_type  # 'Z' denotes a straight line, 'Y' denotes a curve, 'ZH' and 'HZ' are transition curves
        self.seqZ = seqZ  # Length of consecutive straight-line segments
        self.seqY = seqY  # Length of a continuous circular curve
        self.seqS = seqS  # Length of continuous slope
        self.seqSS = seqSS  # Length of continuous slope, but will be reset by a transition curve
        self.start = start  # Whether it is the starting point
        self.pre_ver_cur_len = pre_ver_cur_len  # Half the length of the previous vertical curve
        self.fill_cut_volunm = fill_cut_volunm  # Recording of cut and fill quantities
        self.loc_and_len_cost = loc_and_len_cost  # Costs associated with road length and location
        self.cross_section_cost = cross_section_cost  # Cut and fill costs
        self.get_pre_constrain_inf()  # Get geometric constraint status
        # potential explorations, [[Line type, radius, whether line changes,
        # slope, whether slope changes, whether reset by transition curve],...]
        self.pe = self.get_pe()
        # If the current node is the first node after the vertical curve,
        # get the nodes on the vertical curve before the current node
        # that need to adjust the z-coordinate and the amount of z adjustment
        self.lists_of_ver_curv_nodes_deltazs = self.get_nodes_deltazs()

    def get_nodes_deltazs(self):
        """
        If it is the first node after the vertical curve,
        backtrack forward to get the node information within the vertical curve,
        calculate delta_z and the new slope new_gs
        """
        if self.is_first_out_slope:
            grd_current = self.gradient_z_list[-1]
            # Determine if only one exploration node is within the vertical curve
            not_only_one = False if (self.pre_ver_cur_len <= EXPLORE_RES) else True
            # Difference in z-direction between the traversal point and the vertical curve
            max_deltaz = (self.pre_ver_cur_len**2) / (2 * K_MIN)
            tmp, iter, nodes, deltazs, new_g_tmp = self.pind, 0, [], [], []
            if not_only_one:
                while True:
                    iter = iter + 1
                    # Ratio of distance to half-slope length
                    tmp_ratio = tmp.seqS / self.pre_ver_cur_len
                    new_g_tmp.append(0.5 * tmp_ratio)
                    ratio = 1 - tmp_ratio
                    deltaz = max_deltaz * ratio**2
                    nodes.append(tmp), deltazs.append(deltaz)
                    if tmp.seqS == EXPLORE_RES:
                        # Sign of delta_z, concave is positive, convex is negative
                        sign_z = (
                            1
                            if tmp.gradient_z_list[-1] - tmp.pind.gradient_z_list[-1]
                            > 0
                            else -1
                        )
                        tmp = tmp.pind
                        break
                    tmp = tmp.pind
                # The slope of the traversal point corresponds to the previous slope
                nodes.append(tmp)
                grd_old = tmp.gradient_z_list[-1]
                # delta_z is symmetric
                deltazs = deltazs + [max_deltaz] + deltazs[-1::-1]
                new_g_l, new_g_r = [], []
                for i in new_g_tmp:
                    # Slope similar to delta_z with symmetry
                    new_g_l.append((0.5 - i) * (grd_current - grd_old) + grd_old)
                    new_g_r.insert(0, (0.5 + i) * (grd_current - grd_old) + grd_old)
                    tmp = tmp.pind
                    nodes.append(tmp)
                # The slope at the traversal point is half of the left and right slopes
                new_gs = new_g_l + [0.5 * (grd_current + grd_old)] + new_g_r
                new_gs.reverse()
            else:
                # Only one exploration node lies within a vertical curve
                grd_l, grd_r = tmp.gradient_z_list[-1], self.gradient_z_list[-1]
                sign_z = 1 if grd_r - grd_l > 0 else -1
                nodes.append(tmp), deltazs.append(max_deltaz)
                new_gs = [0.5 * (grd_l + grd_r)]
            return [nodes, deltazs, sign_z, new_gs]
        else:
            return None

    # Constraints before the current node and, since vertical curves are considered,
    # constraints after the current node are also considered
    def get_pre_constrain_inf(self):
        # Continuous slope length constraints
        self.enough_pre_slope_len_vertical = self.seqS >= (
            MIN_LEN_SLOPE + self.pre_ver_cur_len
        )
        # Consider the horizontal vertical retardation disjointness constraint,
        # and since SeqSS <= seqS, it is sufficient to judge only one seqSS here
        self.enough_pre_slope_len_horizontal = self.seqS >= self.pre_ver_cur_len
        # Horizontal straight line length constraints
        self.enough_tangent_len = self.seqZ >= MIN_LEN_TAN
        # Horizontal curve length constraints
        self.enough_curve_len = self.seqY >= MIN_LEN_CURV
        # Determine if the current node is the point at the
        # junction of a vertical curve and the vertical slope.
        self.is_first_out_slope = (
            True
            if (
                (0 <= (self.seqS - self.pre_ver_cur_len) < EXPLORE_RES)
                and (self.pre_ver_cur_len != 0)
            )
            else False
        )

    # To consider vertical curves, consider the constraints after the current node
    def get_next_constrain_inf(self, g_old, g_new):
        next_ver_cur_len = abs(K_MIN * (g_old - g_new)) / 2
        # Overall slope length constraint
        self.enough_next_slope_len_vertical = self.seqS >= (
            MIN_LEN_SLOPE + self.pre_ver_cur_len + next_ver_cur_len
        )
        # Preventing the length of slopes created by the new gradient from
        # coinciding with a transition curve
        self.enough_next_slope_len_horizontal = self.seqSS >= next_ver_cur_len

    def get_pe(self):
        lt = self.line_type
        pe = []  # potential explorations
        grd = self.gradient_z_list[-1]
        if lt == "Z":
            # [[Line type, radius, whether line changes, slope, whether slope
            # changes, whether reset by gentle curve],...]
            if self.seqZ <= MAX_LEN_TAN:
                pe.append(["Z", None, False, grd, False, False])
            # Change slopes on straight sections to meet slope length
            # requirements and no overlap of vertical slopes
            if (self.enough_pre_slope_len_vertical) or (self.start):
                for j in GRADIENTS:
                    change_grd = not (grd == j)
                    if change_grd:
                        self.get_next_constrain_inf(grd, j)
                        if self.start or (
                            self.enough_next_slope_len_vertical
                            and self.enough_next_slope_len_horizontal
                        ):
                            pe.append(["Z", None, False, j, change_grd, False])
            if (self.enough_tangent_len and self.enough_pre_slope_len_horizontal) or (
                self.start
            ):
                for i in RADIUS:
                    pe.append(["ZY", i, True, grd, False, False])
        elif lt == "Y":
            # Prevent backward curves
            if (self.seqY) >= abs(self.radius) * 3.14:
                return pe
            pe.append(["Y", self.radius, False, grd, False, False])
            # Change slopes on circular curves
            if self.enough_pre_slope_len_vertical:
                for j in GRADIENTS:
                    change_grd = not (grd == j)
                    if change_grd:
                        self.get_next_constrain_inf(grd, j)
                        if (
                            self.enough_next_slope_len_vertical
                            and self.enough_next_slope_len_horizontal
                        ):
                            pe.append(["Y", self.radius, False, j, change_grd, False])
            if self.enough_curve_len and self.enough_pre_slope_len_horizontal:
                pe.append(["YZ", self.radius, True, grd, False, False])
        elif lt == "ZY":
            pe.append(["Y", self.radius, True, grd, False, True])
        elif lt == "YZ":
            pe.append(["Z", None, True, grd, False, True])
        return pe


class Config:
    def __init__(self, xw, yw, zw):
        self.xw = xw + 1
        self.yw = yw + 1
        self.zw = zw + 1
        self.theta_xyw = 2 * THETA_XY_IND + 1
        self.gradient_zw = 2 * GRADIENT_Z_IND + 1
        self.minx = 0
        self.miny = 0
        self.minz = 0
        self.min_theta_xy = -THETA_XY_IND
        self.min_gradient_z = -GRADIENT_Z_IND


def get_neighbors(current, config):
    pe = current.pe
    if pe:
        x_old, y_old, z_old = current.xlist[-1], current.ylist[-1], current.zlist[-1]
        theta_xy_old = current.theta_xy_list[-1]
        seqZ, seqY = current.seqZ, current.seqY
        for i in pe:
            # [[Line type, radius, whether line changes, slope, whether slope
            # changes, whether reset by gentle curve],...]
            t, r, c, g, gc, gs = i[0], i[1], i[2], i[3], i[4], i[5]
            # Updating the length of a continuous horizontal curve or horizontal line
            if t == "Z" or t == "Y":  # 计算下一个节点的坐标
                arc_l = EXPLORE_RES  # 此次探索的长度
                seqZ, seqY = 0, 0
                if t == "Z":
                    if c:
                        seqZ = EXPLORE_RES
                    else:
                        seqZ = current.seqZ + arc_l
                elif t == "Y":
                    if c:
                        seqY = EXPLORE_RES
                    else:
                        seqY = current.seqY + arc_l

                x, y, z, theta_xy, gradient_z = tf.z_y_move(
                    x_old, y_old, z_old, theta_xy_old, g, EXPLORE_RES, radi=r
                )
            elif t == "ZY":
                arc_l, seqZ, seqY = LEN_SPIRAL, 0, 0
                x, y, z, theta_xy, gradient_z = tf.spr_move1(
                    x_old, y_old, z_old, theta_xy_old, g, LEN_SPIRAL, LEN_SPIRAL, radi=r
                )
            elif t == "YZ":
                arc_l, seqZ, seqY = LEN_SPIRAL, 0, 0
                x, y, z, theta_xy, gradient_z = tf.spr_move2(
                    x_old, y_old, z_old, theta_xy_old, g, LEN_SPIRAL, LEN_SPIRAL, radi=r
                )
            else:
                raise

            # Updating the length of a continuous slope section and half the
            # length of the previous slope
            (seqS, pre_vc) = (
                (EXPLORE_RES, abs(K_MIN * (current.gradient_z_list[-1] - g)) / 2)
                if gc
                else (current.seqS + arc_l, current.pre_ver_cur_len)
            )
            # The length of the last slope from the starting point is assumed to be zero.
            pre_vc = 0 if current.start else pre_vc
            seqSS = EXPLORE_RES if (gc or gs) else current.seqSS + arc_l

            if x < 0 or x > config.xw or y < 0 or y > config.yw:  # Checking for transgressions
                continue

            is_exceed, cost, c_s_cost, loc_len_cost, f_c_list = get_cost(
                x, y, z, theta_xy, arc_l, g, current
            )
            if is_exceed:  # The slope on the cross-section can not intersect the ground line
                continue

            lens = current.lens + arc_l

            node = Node(
                round(x),
                round(y),
                round(z),
                round(theta_xy / THETA_XY_RESOLUTION),
                round(gradient_z / GRADIENT_Z_RESOLUTION),
                [x],
                [y],
                [z],
                [theta_xy],
                [gradient_z],
                line_type=t,
                cost=cost,
                radius=r,
                pind=current,
                seqY=seqY,
                seqZ=seqZ,
                seqS=seqS,
                seqSS=seqSS,
                pre_ver_cur_len=pre_vc,
                lens=lens,
                fill_cut_volunm=f_c_list,
                loc_and_len_cost=loc_len_cost,
                cross_section_cost=c_s_cost,
            )

            yield node


def get_cost(x, y, z, a, s, g, current):
    # If it is the first node after a vertical curve, we need to backtrack
    # forward to recalculate the cut and fill, etc.
    if current.is_first_out_slope:  
        [nodes, deltazs, sign_z, new_gs] = current.lists_of_ver_curv_nodes_deltazs
        f_c_list_tmp = copy.deepcopy(nodes[-1].pind.fill_cut_volunm)
        loc_len_cos_tmp = copy.deepcopy(nodes[-1].pind.loc_and_len_cost)
        for node, dz, new_g in zip(nodes + [current], deltazs + [0], new_gs + [g]):
            # Update new z-coordinates for each exploration node
            tx, ty, tz, ta = (
                node.xlist[-1],
                node.ylist[-1],
                node.zlist[-1] + sign_z * dz,
                node.theta_xy_list[-1],
            )
            # The vertical curves are all straight or circular exploration
            # nodes, so the exploration lengths are all EXPLORE_RES
            f_c_list_in, loc_len_cos_in, is_exceed = calculate_cost(
                f_c_list_tmp, loc_len_cos_tmp, tx, ty, tz, ta, EXPLORE_RES, new_g
            )
            if is_exceed:
                return (is_exceed, None, None, None, None)
            f_c_list_tmp, loc_len_cos_tmp = f_c_list_in, loc_len_cos_in
    else:
        f_c_list_in = current.fill_cut_volunm
        loc_len_cos_in = current.loc_and_len_cost

    f_c_list, loc_len_cos, is_exceed = calculate_cost(
        f_c_list_in, loc_len_cos_in, x, y, z, a, s, g
    )
    if is_exceed:
        return (is_exceed, None, None, None, None)
    # Calculate cross-section cut and fill costs
    cross_section_cost = get_multi_layer_cost(f_c_list[2], f_c_list[3], f_c_list[1])
    # Calculate total cost
    total_cost = sum(cross_section_cost) + sum(loc_len_cos)
    return (is_exceed, total_cost, cross_section_cost, loc_len_cos, f_c_list)


def calculate_cost(f_c_list_in, loc_len_cost_in, x, y, z, a, s, g):
    [tf1_0, tf2_0, tc1_0, tc2_0] = f_c_list_in
    [lc0, cc0, pc0, mc0] = loc_len_cost_in
    # Calculate the amount of excavation and filling corresponding to the two
    # layers of soil, the upper layer being soil layer "2".
    tmp_f2, tmp_c2, tmp_wid, is_exceed = get_cross_fill_cut(
        x, y, z, a, terrain, 7.58, 25, math.tan(0.78), math.tan(0.59)
    )
    if is_exceed:
        return (None, None, is_exceed)
    tmp_f1, tmp_c1, _, _ = get_cross_fill_cut(
        x, y, z, a, terrain1, 7.58, 25, math.tan(0.78), math.tan(0.59)
    )
    tf1_0, tf2_0, tc1_0, tc2_0 = (
        tf1_0 + tmp_f1 * s,
        tf2_0 + tmp_f2 * s,
        tc1_0 + tmp_c1 * s,
        tc2_0 + tmp_c2 * s,
    )
    # Calculate the cost of acreage, clearing, paving and maintenance.
    lc0 = lc0 + (tmp_wid + 6) * 7 * s
    cc0 = cc0 + 0.6 * tmp_wid * s * (1 + g**2) ** 0.5
    pc0 = pc0 + 550 * s * (1 + g**2) ** 0.5
    mc0 = mc0 + 100 * s * (1 + g**2) ** 0.5
    f_c_list = [tf1_0, tf2_0, tc1_0, tc2_0]
    loc_len_cos = [lc0, cc0, pc0, mc0]
    return (f_c_list, loc_len_cos, is_exceed)


def distance_point_to_line(A, B):
    """
    A is a tuple containing the x, y coordinates and azimuth of vector A. B is
    a tuple containing the x, y coordinates of point B. The function returns
    the distance from the vertical line of the point B to the vector A
    """
    x1, y1, angle = A
    x2, y2 = B
    k = math.tan(angle)
    if k == 0:
        return abs(y2 - y1)
    else:
        k2 = -1 / k
        b1 = y1 - k * x1
        b2 = y2 - k2 * x2
        x = (b2 - b1) / (k - k2)
        y = k * x + b1
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)


def distance_point_to_point(A, B):
    x1, y1 = A
    x2, y2 = B
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def intersection(x1, y1, radian1, x2, y2, radian2):
    # Calculate the coordinates of the intersection point
    # Calculate the direction vector
    v1 = np.array([np.cos(radian1), np.sin(radian1)])
    v2 = np.array([np.cos(radian2), np.sin(radian2)])
    # Calculate point coordinates
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])

    cross = np.cross(v2, v1)

    t = np.cross(p1 - p2, v2) / cross
    inter = p1 + t * v1
    return inter

"""
main file part
"""
# endpoint
(xw, yw) = terrain.shape
goal = [4000, 2000, terrain[4000][2000], tf.pi_2_pi(np.deg2rad(0))]
ngoal = Node(
    round(goal[0]),
    round(goal[1]),
    round(goal[2]),
    round(goal[3] / THETA_XY_RESOLUTION),
    0,
    [goal[0]],
    [goal[1]],
    [goal[2]],
    [goal[3]],
)
# Here z needs to be greater than the maximum height difference in the study
# area to avoid misclassification in the calculation of the nodal metrics.
config = Config(xw, yw, 1000)
openList, closedList = {}, {}
pq = []
for i in range(13):
    # start point with angles [-90, -75, ..., 90]
    start = [250, 1000, terrain[250][1000], tf.pi_2_pi(np.deg2rad(-90 + 15 * i))]
    nstart = Node(
        round(start[0]),
        round(start[1]),
        round(start[2]),
        round(start[3] / THETA_XY_RESOLUTION),
        0,
        [start[0]],
        [start[1]],
        [start[2]],
        [start[3]],
        start=True,
        line_type="Z",
        fill_cut_volunm=[0, 0, 0, 0],
        loc_and_len_cost=[0, 0, 0, 0],
    )
    openList[tf.calc_index(nstart, config)] = nstart
    heapq.heappush(
        pq, (tf.calc_cost_new(nstart, heuristic), tf.calc_index(nstart, config))
    )
num_iter = 0

while True:
    if not openList:
        print("Error: Cannot find path, No open set")
        break

    cost, c_id = heapq.heappop(pq)
    if c_id in openList:
        current = openList.pop(c_id)
        closedList[c_id] = current
    else:
        continue

    num_iter += 1
    if num_iter % 50000 == 0:
        print(num_iter)

    if (
        abs(current.xind - ngoal.xind) + abs(current.yind - ngoal.yind)
        < 2 * EXPLORE_RES
    ):
        cx, cy, cz, ctheta, cg = (
            current.xlist[-1],
            current.ylist[-1],
            current.zlist[-1],
            current.theta_xy_list[-1],
            current.gradient_z_list[-1],
        )
        gx, gy, gz = goal[0], goal[1], goal[2]
        dis_points = distance_point_to_point((cx, cy), (gx, gy))
        dis_p_to_l = distance_point_to_line((cx, cy, ctheta), (gx, gy))
        delta_z = abs(cz + dis_points * cg - gz)
        if (
            (current.line_type == "Z")
            and (dis_points <= EXPLORE_RES)
            and (dis_p_to_l <= 1)
            and (current.enough_pre_slope_len_vertical)
            and (delta_z < 0.4)
        ):
            is_exceed, cost, c_s_cost, loc_len_cost, f_c_list = get_cost(
                gx, gy, gz, ctheta, dis_points, cg, current
            )
            print("delta z is {} m".format(delta_z))
            print("cuurent total cost is {}".format(cost))
            print("cuurent cross_section_cost cost is {}".format(c_s_cost))
            print("cuurent loc_and_len_cost cost is {}".format(loc_len_cost))
            break

    for neighbor in get_neighbors(current, config):
        neighbor_index = tf.calc_index(neighbor, config)
        if neighbor_index in closedList:
            continue
        if neighbor not in openList or openList[neighbor_index].cost > neighbor.cost:
            heapq.heappush(pq, (tf.calc_cost_new(neighbor, heuristic), neighbor_index))
            openList[neighbor_index] = neighbor

# End point angle rotation
tmp0 = current
while tmp0.line_type != "Y":
    tmp0 = tmp0.pind
tx, ty, ttheta, tr = tmp0.xlist[-1], tmp0.ylist[-1], tmp0.theta_xy_list[-1], tmp0.radius
ox, oy = tx - tr * math.sin(ttheta), ty + tr * math.cos(ttheta)
p_to_l = distance_point_to_line((cx, cy, ctheta), (ox, oy))
p_to_p = distance_point_to_point((ox, oy), (gx, gy))
angle_goal_to_center = math.atan((gy - oy) / (gx - ox))
d_theta = math.asin(p_to_l / p_to_p)
# Angle after endpoint rotation
angle_goal = angle_goal_to_center + d_theta * np.sign(tr)
if abs(ctheta - angle_goal) > math.pi / 2:
    angle_goal = angle_goal + math.pi
# The angle of rotation of the centre angle of the circle before the end point,
# positive is to increase the length of the arc, negative is to decrease the
# length of the arc
rotate_angle = (angle_goal - ctheta) * np.sign(tr)

# Find the coordinates of the location of the horizontal intersection and the
# mileage of the traversal points.
pi_x, pi_y, pi_r, pi_w = [gx], [gy], [0], [0]
old_x, old_y, old_angle = gx, gy, angle_goal
bpd_lens, bpd_gs = [], []
tmp = current
while tmp.pind is not None:
    if tmp.line_type == "YZ":
        tmp_r, cur_len = tmp.pind.radius, tmp.pind.seqY
        pi_r = [tmp_r] + pi_r
        pi_w = [abs(cur_len / tmp_r)] + pi_w
    if tmp.line_type == "ZY":
        new_x, new_y = tmp.pind.xlist[-1], tmp.pind.ylist[-1]
        new_angle = tmp.pind.theta_xy_list[-1]
        [x_t, y_t] = intersection(old_x, old_y, old_angle, new_x, new_y, new_angle)
        pi_x = [x_t] + pi_x
        pi_y = [y_t] + pi_y
        old_x, old_y, old_angle = new_x, new_y, new_angle
    if tmp.gradient_z_list[-1] != tmp.pind.gradient_z_list[-1]:
        bpd_gs = tmp.pind.gradient_z_list + bpd_gs
        bpd_lens = [tmp.pind.lens] + bpd_lens
    tmp = tmp.pind
pi_w[-2] = pi_w[-2] + rotate_angle
pi_x.insert(0, start[0]), pi_y.insert(0, start[1]), pi_r.insert(0, 0), pi_w.insert(0, 0)
out_points = []
for x, y, r, w in zip(pi_x, pi_y, pi_r, pi_w):
    out_points += [x / 1000, y / 1000, r / 1000, abs(w)]
np.savetxt("points.txt", np.array(out_points))
np.savetxt("slopes.txt", np.array(bpd_gs[1:]))
np.savetxt("bpd_lens.txt", np.array(bpd_lens[1:]) / 1000)
plt.plot(pi_x, pi_y, ".-y")

# Plot the scatterplot of the x-y coordinates of the exploration nodes
tmp = current
while tmp.pind is not None:
    if tmp.line_type == "Z" or tmp.line_type == "YZ":
        plt.plot(tmp.xlist[-1], tmp.ylist[-1], ".r")
    else:
        plt.plot(tmp.xlist[-1], tmp.ylist[-1], ".g")
    tmp = tmp.pind
plt.plot(tmp.xlist[-1], tmp.ylist[-1], ".r")
plt.plot(start[0], start[1], "*r")
plt.plot(goal[0], goal[1], "*r")
plt.axis("equal")
plt.figure()

# Plot the vertical alignment with the horizontal axis as the mileage
tmp = current
px, pz, pt = [], [], []
while tmp.pind is not None:
    px.append(tmp.lens), pz.append(tmp.zlist[-1]), pt.append(
        terrain[tmp.xind][tmp.yind]
    )
    if tmp.gradient_z_list[-1] != tmp.pind.gradient_z_list[-1]:
        plt.plot(tmp.pind.lens, tmp.pind.zlist[-1], "oy")
    if tmp.is_first_out_slope:
        [nodes, deltazs, sign_z, new_gs] = tmp.lists_of_ver_curv_nodes_deltazs
        for i in range(len(deltazs)):
            tmp_x, tmp_y = nodes[i].lens, nodes[i].zlist[-1] + sign_z * deltazs[i]
            plt.plot(tmp_x, tmp_y, "oc")
            plt.plot(
                [tmp_x, tmp_x + EXPLORE_RES],
                [tmp_y, tmp_y + EXPLORE_RES * new_gs[i]],
                "-c",
            )
    tmp = tmp.pind
px.append(tmp.lens), pz.append(tmp.zlist[-1]), pt.append(terrain[tmp.xind][tmp.yind])
plt.plot(px, pz, "-r")
plt.plot(px, pt, "-.k")
plt.show()