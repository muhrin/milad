# -*- coding: utf-8 -*-
import math
import numba
import numpy as np


def get_num_plane_repetitions_to_bound_sphere(
    radius: float, volume: float, cross_len: float
) -> float:
    # The vector normal to the plane
    return radius / volume * cross_len


class AperiodicDistanceCalculator:
    def get_vecs_between(
        self,
        a: np.array,
        b: np.array,
        cutoff: float,
        max_cell_multiples: int = 100000,
        self_interation=True,
    ) -> np.array:  # pylint: disable=invalid-name
        dr = b - a
        cutoff_sq = cutoff * cutoff

        out_values = []
        out_vec = dr
        if self_interation and np.dot(out_vec, out_vec) < cutoff_sq:
            out_values.append(out_vec)

        return np.array(out_values)


class UnitCellDistanceCalculator:
    def __init__(self, unit_cell: np.array):
        # Precompute some useful values
        self._cell = unit_cell
        self._a = self._cell[0]
        self._b = self._cell[1]
        self._c = self._cell[2]
        self._volume = np.abs(np.dot(self._a, np.cross(self._b, self._c)))

        self._a_cross_b = np.cross(self._a, self._b)
        self._a_cross_b_len = np.linalg.norm(self._a_cross_b)
        self._a_cross_b_hat = self._a_cross_b / self._a_cross_b_len

        self._b_cross_c = np.cross(self._b, self._c)
        self._b_cross_c_len = np.linalg.norm(self._b_cross_c)
        self._b_cross_c_hat = self._b_cross_c / self._b_cross_c_len

        self._a_cross_c = np.cross(self._a, self._c)
        self._a_cross_c_len = np.linalg.norm(self._a_cross_c)
        self._a_cross_c_hat = self._a_cross_c / self._a_cross_c_len

    def get_num_plane_repetitions_to_bound_sphere_planes(
        self, plane_vec1: np.array, plane_vec2: np.array, radius: float
    ) -> float:
        # The vector normal to the plane
        normal = np.cross(plane_vec1, plane_vec2)
        vol = self._volume  # = a . |b x c|
        return radius / vol * np.linalg.norm(normal)

    def get_vec_min_img(
        self,
        a: np.array,
        b: np.array,
        max_cell_multiples: int = 100000,
        self_interation=True,
    ) -> np.array:  # pylint: disable=invalid-name
        """
        Get all vectors from a to b that are less than the cutoff in length

        :param a:
        :param b:
        :param cutoff:
        :param max_cell_multiples:
        :param self_interation:
        :return:
        """
        vol = self._volume
        # TODO: Wrap a, and b into the current unit cell
        dr = b - a

        min_dr_sq = np.dot(dr, dr)
        min_length = min_dr_sq**0.5

        a_max = math.ceil(
            get_num_plane_repetitions_to_bound_sphere(
                min_length, vol, self._b_cross_c_len
            )
        )

        b_max = math.ceil(
            get_num_plane_repetitions_to_bound_sphere(
                min_length, vol, self._a_cross_c_len
            )
        )

        c_max = math.ceil(
            get_num_plane_repetitions_to_bound_sphere(
                min_length, vol, self._a_cross_b_len
            )
        )

        a_max = min(a_max, max_cell_multiples)
        b_max = min(b_max, max_cell_multiples)
        c_max = min(c_max, max_cell_multiples)

        # min_dr = fast_min_img(dr, self._a, self._b, self._c, a_max, b_max, c_max, self_interation)
        # return min_dr

        min_dr = dr

        for i in range(-a_max, a_max + 1):
            ra = i * self._a
            for j in range(-b_max, b_max + 1):
                rab = ra + j * self._b
                for k in range(-c_max, c_max + 1):
                    if not self_interation and i == 0 and j == 0 and k == 0:
                        continue

                    out_vec = rab + k * self._c + dr
                    len_sq = np.dot(out_vec, out_vec)
                    if len_sq < min_dr_sq:
                        min_dr = out_vec
                        min_dr_sq = len_sq

        return min_dr

    def get_vecs_between(
        self,
        a: np.array,
        b: np.array,
        cutoff: float,
        max_cell_multiples: int = 100000,
        self_interation=True,
    ) -> np.array:  # pylint: disable=invalid-name
        """
        Get all vectors from a to b that are less than the cutoff in length

        :param a:
        :param b:
        :param cutoff:
        :param max_cell_multiples:
        :param self_interation:
        :return:
        """
        vol = self._volume
        # TODO: Wrap a, and b into the current unit cell
        dr = b - a

        a_max = math.floor(
            get_num_plane_repetitions_to_bound_sphere(
                cutoff + math.fabs(np.dot(dr, self._b_cross_c_hat)),
                vol,
                self._b_cross_c_len,
            )
        )

        b_max = math.floor(
            get_num_plane_repetitions_to_bound_sphere(
                cutoff + math.fabs(np.dot(dr, self._a_cross_c_hat)),
                vol,
                self._a_cross_c_len,
            )
        )

        c_max = math.floor(
            get_num_plane_repetitions_to_bound_sphere(
                cutoff + math.fabs(np.dot(dr, self._a_cross_b_hat)),
                vol,
                self._a_cross_b_len,
            )
        )

        a_max = min(a_max, max_cell_multiples)
        b_max = min(b_max, max_cell_multiples)
        c_max = min(c_max, max_cell_multiples)

        cutoff_sq = cutoff * cutoff
        out_values = []

        for i in range(-a_max, a_max + 1):
            ra = i * self._a
            for j in range(-b_max, b_max + 1):
                rab = ra + j * self._b
                for k in range(-c_max, c_max + 1):
                    if not self_interation and i == 0 and j == 0 and k == 0:
                        continue

                    out_vec = rab + k * self._c + dr
                    if np.dot(out_vec, out_vec) < cutoff_sq:
                        out_values.append(out_vec)

        return np.array(out_values)


@numba.jit(parallel=True)
def fast_min_img(dr, a, b, c, a_max, b_max, c_max, self_interation):
    min_dr = dr.copy()
    min_dr_sq = np.dot(dr, dr)

    for i in range(-a_max, a_max + 1):
        ra = i * a
        for j in range(-b_max, b_max + 1):
            rab = ra + j * b
            for k in range(-c_max, c_max + 1):
                if not self_interation and i == 0 and j == 0 and k == 0:
                    continue

                out_vec = rab + k * c + dr
                len_sq = np.dot(out_vec, out_vec)
                if len_sq < min_dr_sq:
                    min_dr = out_vec
                    min_dr_sq = len_sq

    return min_dr
