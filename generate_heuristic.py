"""

improved A* grid based planning

author: Tianlong Zhang (SWJTU)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import heapq
import matplotlib.pyplot as plt
import numpy as np

terrain = np.loadtxt("terrain.txt") * 1000
show_animation = False


class Node:
    def __init__(self, x, y, cost, pind, z=0):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind
        self.z = z

    def __str__(self):
        return (
            str(self.x)
            + ","
            + str(self.y)
            + ","
            + str(self.cost)
            + ","
            + str(self.pind)
        )


def calc_final_path(ngoal, closedset, reso):
    # generate final course
    rx, ry = [ngoal.x * reso], [ngoal.y * reso]
    pind = ngoal.pind
    if show_animation:
        print("total cost is {}".format(ngoal.cost))
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        pind = n.pind

    return rx, ry


def dp_planning(sx, sy, gx, gy, ox, oy, reso, rr, max_g):

    nstart = Node(round(sx), round(sy), 0.0, -1)
    ngoal = Node(round(gx), round(gy), 0.0, -1)
    ngoal.z = terrain[round(gx), round(gy)]

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, rr)

    motion = get_motion_model()

    openset, closedset = dict(), dict()
    openset[calc_index(ngoal, xw, minx, miny)] = ngoal
    pq = []
    pq.append((0, calc_index(ngoal, xw, minx, miny)))

    Z = np.zeros(terrain.shape)
    while 1:
        if not pq:
            break
        cost, c_id = heapq.heappop(pq)
        if c_id in openset:
            current = openset[c_id]
            closedset[c_id] = current
            openset.pop(c_id)
        else:
            continue
        Z[current.y, current.x] = cost
        # show graph
        if show_animation:  # pragma: no cover
            if len(closedset.keys()) % 5000000 == 0:
                plt.plot(current.x * reso, current.y * reso, "xc")
            if len(closedset.keys()) % 5000000 == 0:
                plt.pause(0.001)

        # Remove the item from the open set

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            new_x = current.x + motion[i][0]
            new_y = current.y + motion[i][1]
            in_area = (0 <= new_x < maxx - 1) and (0 <= new_y < maxy - 1)
            if not in_area:
                continue
            # whether the terrain exceeds the maximum slope and whether the z-coordinate is an accessible elevation
            ter_dif = terrain[new_x, new_y] - current.z
            abs_td = abs(ter_dif)
            sign_td = 1 if ter_dif > 0 else -1
            # Consider the maximum slope and count cut and fill above it
            if abs_td > motion[i][2] * max_g:
                new_z = current.z + sign_td * motion[i][2] * max_g
                csc = (max_g**2 + 1**2) ** 0.5  # projection ratio
            else:
                new_z = terrain[new_x, new_y]
                csc = ((new_z - current.z) ** 2 + (motion[i][2]) ** 2) ** 0.5 / (
                    motion[i][2]
                )  # projection ratio
            # approximate cut and fill cost
            volumn = (
                abs(
                    (terrain[new_x, new_y] - new_z)
                    + (terrain[current.x, current.y] - current.z)
                )
                * motion[i][2]
                / 2
                * 15.15
            )
            v_price = volumn * 20
            new_cost = (
                current.cost
                + motion[i][2] * 660 * csc
                + motion[i][2] * csc * (15.15 + 6) * 7
                + v_price
            )
            node = Node(new_x, new_y, new_cost, c_id, z=new_z)
            n_id = calc_index(node, xw, minx, miny)

            if n_id in closedset:
                continue

            if not verify_node(node, obmap, minx, miny, maxx, maxy):
                continue

            if n_id not in openset:
                openset[n_id] = node  # Discover a new node
                heapq.heappush(pq, (node.cost, calc_index(node, xw, minx, miny)))
            else:
                if openset[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    openset[n_id] = node
                    heapq.heappush(pq, (node.cost, calc_index(node, xw, minx, miny)))

    rx, ry = calc_final_path(
        closedset[calc_index(nstart, xw, minx, miny)], closedset, reso
    )

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        # artists.append(frame)
        plt.pause(0.001)

    return rx, ry, closedset, Z


def calc_heuristic(n1, n2):
    w = 1.0  # weight of heuristic
    d = w * math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)
    return d


def verify_node(node, obmap, minx, miny, maxx, maxy):

    if node.x < minx:
        return False
    elif node.y < miny:
        return False
    elif node.x >= maxx:
        return False
    elif node.y >= maxy:
        return False

    if obmap[node.x][node.y]:
        return False

    return True


def calc_obstacle_map(ox, oy, reso, vr):

    minx = 0
    miny = 0
    maxx = max(ox) + 1
    maxy = max(oy) + 1

    xwidth = maxx
    ywidth = maxy

    # obstacle map generation
    obmap = [[False for i in range(ywidth)] for i in range(xwidth)]
    for i in range(len(ox)):
        obmap[ox[i]][oy[i]] = True

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)


def get_motion_model():
    # dx, dy, cost
    motion = [
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, -1, 1],
        [-1, -1, math.sqrt(2)],
        [-1, 1, math.sqrt(2)],
        [1, -1, math.sqrt(2)],
        [1, 1, math.sqrt(2)],
    ]

    return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    global frame
    global artists
    sx = 250.0  # [m]
    sy = 1000.0  # [m]
    gx = 4000.0  # [m]
    gy = 2000.0  # [m]
    grid_size = 1.0  # [m]
    robot_size = 1.5  # [m]

    max_gradient = 0.05

    ox, oy = [], []
    (xw, yw) = terrain.shape
    for i in range(xw + 1):
        ox.append(i), oy.append(0), ox.append(i), oy.append(yw)
    for i in range(yw + 1):
        ox.append(0), oy.append(i), ox.append(xw), oy.append(i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "xr")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    rx, ry, closedset, Z = dp_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_size, max_gradient
    )
    plt.plot(rx, ry, "-r")
    np.savetxt('heuristic.txt', Z, fmt='%d')
    plt.show()


if __name__ == "__main__":
    show_animation = True
    main()
    raise
