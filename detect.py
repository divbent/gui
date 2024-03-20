from ultralytics import YOLO
import os
import cv2 as cv
import math
import numpy as np
import random


def make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
                left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0],
                                                                                            left[1])):
            left = c
        elif cross < 0.0 and (
                right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0],
                                                                                             right[1])):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_diameter(a, b):
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    r0 = math.hypot(cx - a[0], cy - a[1])
    r1 = math.hypot(cx - b[0], cy - b[1])
    return (cx, cy, max(r0, r1))


def make_circumcircle(a, b, c):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
    ax = a[0] - ox;
    ay = a[1] - oy
    bx = b[0] - ox;
    by = b[1] - oy
    cx = c[0] - ox;
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def make_diameter(a, b):
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    r0 = math.hypot(cx - a[0], cy - a[1])
    r1 = math.hypot(cx - b[0], cy - b[1])
    return (cx, cy, max(r0, r1))

def predict(img, confidence, st):
    lenx=500
    leny=500
    target_centerx = 0
    target_centery = 0
    otklonenie = 0
    centers = []
    model_path = os.path.join('.', 'best.pt')
    model = YOLO(model_path)
    results = model.predict(img, conf=confidence)
    result = results[0]

    for obj in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = obj
        # print("x = ", x1, "y = ", y1, "x = ", x2, "y = ", y2, "class_id = ", class_id)
        if class_id == 2:
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 4)
            lenx = x2 - x1
            leny = y2 - y1
            target_centerx = (x2 + x1) / 2
            target_centery = (y2 + y1) / 2
        else:
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            centers.append([(x1 + x2)/2, (y1 + y2)/2])

    points = np.array(centers)
    centerx, centery, radius = make_circle(points)
    centerx = int(centerx)
    centery = int(centery)
    cv.circle(img, (centerx, centery), math.ceil(radius), (255, 2, 2), 5)
    tarelka = (20 * radius) / (lenx + leny)
    for center in centers:
        otklonenie += math.sqrt(pow((((target_centerx - center[0]) / lenx) * 10), 2) + pow((((target_centery - center[1]) / leny) * 10), 2))
    otklonenie = otklonenie / len(centers)

    st.subheader('Output Image')
    small_img = cv.resize(img, (1000, int(1000 * img.shape[0] / img.shape[1])))
    st.image(small_img, channels="BGR", use_column_width=False)
    st.write(f"все попадания находяться в окружности радиуса {tarelka} см")
    st.write(f"cреднее отклонение от СТП {otklonenie} см")
