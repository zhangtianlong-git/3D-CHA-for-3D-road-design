"""

get fill, cut and corresponding cost information

author: Tianlong Zhang (SWJTU)

"""
import numpy as np
import math
from matplotlib import pyplot as plt


def get_cross_fill_cut(
    x, y, z, theta_xy, terrain, roadhalf, off_road, slop_g_c, slope_g_f
):
    """
    The general idea is to divide the cross-section into three parts, left,
    centre and right, then calculate the corresponding height difference,
    and finally merge the matrices to find the amount of excavation and filling.
    """
    deltax, deltay = math.cos(theta_xy - math.pi / 2), math.sin(theta_xy - math.pi / 2)
    d_r = terrain[round(x + roadhalf * deltax), round(y + roadhalf * deltay)] - z
    d_l = terrain[round(x - roadhalf * deltax), round(y - roadhalf * deltay)] - z
    sign_d_r, sign_d_l = np.sign(d_r), np.sign(d_l)
    ## Whether left or right cut or fill to determine slope
    slop_g_r = slop_g_c if d_r > 0 else slope_g_f
    slop_g_l = slop_g_c if d_l > 0 else slope_g_f
    tmp = np.arange(1, int(off_road) + 1)
    tmp_arr = tmp + roadhalf
    xs_r, xs_l = x + tmp_arr * deltax, x - tmp_arr * deltax
    ys_r, ys_l = y + tmp_arr * deltay, y - tmp_arr * deltay
    ## zs are the vertical coordinates on the left and right slopes of the cross section
    zs_r, zs_l = z + sign_d_r * tmp * slop_g_r, z + sign_d_l * tmp * slop_g_l
    zs_r_delta = terrain[np.round(xs_r).astype(int), np.round(ys_r).astype(int)] - zs_r
    zs_l_delta = terrain[np.round(xs_l).astype(int), np.round(ys_l).astype(int)] - zs_l
    ## Here index characterises the intersection of the ground line and the cross-section,
    ## and must be used greater than 0 to prevent indexing errors
    r_index, l_index = zs_r_delta * sign_d_r > 0, zs_l_delta * sign_d_l > 0
    zs_r_delta_final, zs_l_delta_final = zs_r_delta[r_index], zs_l_delta[l_index]
    is_exceed = False
    if len(zs_r_delta_final) == int(off_road) or len(zs_l_delta_final) == int(off_road):
        is_exceed = True
    ## Here the corresponding height difference within the width of the road is counted, and all are valid.
    delta_int_road = roadhalf - int(roadhalf)
    tmp1 = np.arange(int(-roadhalf), int(roadhalf) + 1)
    xs_c, ys_c = x + tmp1 * deltax, y + tmp1 * deltay
    zs_c_delta_final = (
        terrain[np.round(xs_c).astype(int), np.round(ys_c).astype(int)] - z
    )
    ## Merge matrix of effective height differences
    z_delta_final = np.concatenate(
        (zs_l_delta_final, zs_c_delta_final, zs_r_delta_final)
    )
    fill, excave = sum(z_delta_final[z_delta_final < 0]), sum(
        z_delta_final[z_delta_final >= 0]
    )
    ## Updating of the cut and fill and site boundaries for the rounded portion of the road
    tmp_list = delta_int_road * np.array([zs_c_delta_final[0], zs_c_delta_final[-1]])
    fill += sum(tmp_list[tmp_list < 0])
    excave += sum(tmp_list[tmp_list > 0])
    width_l, width_r = (
        len(zs_l_delta_final) + roadhalf,
        len(zs_r_delta_final) + roadhalf,
    )
    width = width_l + width_r
    tmp_list1 = []
    tmp_list1 = (
        tmp_list1 + [zs_r_delta_final[-1]]
        if len(zs_r_delta_final) > 0
        else tmp_list1 + [zs_c_delta_final[-1]]
    )
    tmp_list1 = (
        tmp_list1 + [zs_l_delta_final[-1]]
        if len(zs_l_delta_final) > 0
        else tmp_list1 + [zs_c_delta_final[0]]
    )
    tmp_list2 = np.array([tmp_list1])
    width += abs(sum(tmp_list2[tmp_list2 < 0] / slope_g_f))
    width += abs(sum(tmp_list2[tmp_list2 > 0] / slop_g_c))
    if __name__ == "__main__":
        xs_r_final, ys_r_final, zs_r_final = xs_r[r_index], ys_r[r_index], zs_r[r_index]
        xs_l_final, ys_l_final, zs_l_final = xs_l[l_index], ys_l[l_index], zs_l[l_index]
        z_final = np.concatenate((zs_l_final, [z] * len(tmp1), zs_r_final))
        x_final = np.concatenate((xs_l_final, xs_c, xs_r_final))
        y_final = np.concatenate((ys_l_final, ys_c, ys_r_final))
        t_final = terrain[x_final.astype(int), y_final.astype(int)]
        return (
            abs(fill),
            abs(excave),
            [x_final, y_final, z_final, t_final],
            width_l,
            width_r,
            is_exceed,
        )
    return (abs(fill), abs(excave), width, is_exceed)


def get_multi_layer_cost(cut1, cut2, fill2):
    """
    terrain with two layers
    See article (https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12350)
    """
    p1, p2, pr, pb, pw, c1, c2, r1, r2, s1, s2 = (
        5,
        2,
        1.1,
        4.5,
        2.0,
        0.97,
        0.95,
        1,
        0.8,
        1.25,
        1.25,
    )
    vc1, vc2, vf = cut1, cut2 - cut1, fill2
    vr = min(vf, vc1 * c1 * r1 + vc2 * c2 * r2)
    vb = vf - vr
    vw = (
        (1 - c1) * s1 * vc1
        + (1 - c2) * s2 * vc2
        + (1.25 / 0.9) * (vc1 * c1 * r1 + vc2 * c2 * r2 - vr)
    )
    return [p1 * vc1 + p2 * vc2, pr * vr, pb * vb, pw * vw]


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    X, Y = np.meshgrid(np.arange(150), np.arange(160))
    Z = (X + Y) * 0.5
    (
        fill,
        excave,
        [x_final, y_final, z_final, t_final],
        width_l,
        width_r,
        is_exceed,
    ) = get_cross_fill_cut(50, 50, 50, math.pi, Z, 7.58, 25, 1 / 0.5, 1 / 1.5)
    print(
        "fill is {}, and cut is {}, width_l is {}, width_r is {}, is exceed is {}".format(
            fill, excave, width_l, width_r, is_exceed
        )
    )
    ax.plot(x_final, y_final, z_final, ".r")
    ax.plot(x_final, y_final, t_final, ".k")
    plt.show()
