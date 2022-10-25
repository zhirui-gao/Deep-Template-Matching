import random

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
import cv2

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 200)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)

class Curves():
    def __init__(self, n):
        np.random.seed(10)
        self.rad = 0.2
        self.edgy = np.random.rand()
        self.n = n

    def get_point(self,min_x, max_x, min_y, max_y):
        a = self.get_random_points(n=self.n, scale=1)
        x, y, _ = self.get_bezier_curve(a, rad=self.rad, edgy=self.edgy)
        x = x * (max_x-min_x) + min_x
        y = y * (max_y-min_y) + min_y
        points = np.array(np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)) # N,2
        return points


    def get_curve(self, points, **kw):
        segments = []
        for i in range(len(points) - 1):
            seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def ccw_sort(self, p):
        d = p - np.mean(p, axis=0)
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]

    def get_bezier_curve(self,a, rad=0.2, edgy=0):
        """ given an array of points *a*, create a curve through
        those points.
        *rad* is a number between 0 and 1 to steer the distance of
              control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
               edgy=0 is smoothest."""
        p = np.arctan(edgy) / np.pi + .5
        a = self.ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
        d = np.diff(a, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
        ang = f(ang)
        ang1 = ang
        ang2 = np.roll(ang, 1)
        ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
        ang = np.append(ang, [ang[0]])
        a = np.append(a, np.atleast_2d(ang).T, axis=1)
        s, c = self.get_curve(a, r=rad, method="var")
        x, y = c.T
        return x, y, a

    def get_random_points(self, n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7 / n
        a = np.random.rand(n, 2)
        d = np.sqrt(np.sum(np.diff(self.ccw_sort(a), axis=0), axis=1) ** 2)
        if np.all(d >= mindst) or rec >= 200:
            return a * scale
        else:
            return self.get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)

    def draw_plot(self,x, y):
        img = np.zeros((480, 640), np.uint8)
        for i in range(len(x) - 1):
            cv2.line(img, (int(x[i]), int(y[i])), (int(x[i + 1]), int(y[i + 1])), 255, 1, 8)
        cv2.imwrite('1.png', img)



# r = Curves(5,min_x=10,max_x=630,min_y=10,max_y=470)









# plt.show()