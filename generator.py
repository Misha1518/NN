import numpy as np
import random as r


class Generator():
    def __init__(self, dimensions=2, classes=2, classdots=250, noise=0, intersections=0, borders=10, json=None):

        if json is None:
            self.dimensions           = dimensions
            self.classes        = classes
            self.classdots      = classdots
            self.noise          = noise
            self.intersections  = intersections
            self.borders         = borders

            self.data           = None

        else:
            self.from_json(json)

    def generate(self):

        if (
            self.dimensions < 2 or self.dimensions > 30
        ):
            print('error')
            return

        if (
            self.classes < 2 or self.classes > 100
        ):
            print('error')
            return

        self.data = generator(dimensions=self.dimensions, classes=self.classes, classdots=self.classdots, noise=self.noise, intersections=self.intersections, borders=self.borders)

        return self.data

    def plot(self, show=True):
        if not self.data:
            print('    No data to plot!')
            return

        if self.dimensions > 2:
            print('Can only draw up to 3 dimensions')
            return

        plot(dz=self.data['dots'], borders=self.borders, show=show)

        return

    def csv(self, directory, test=0.3, val=0):
        import csv
        from os.path import join

        dots = self.data['dots']

        classtest_dots  = int(self.classdots * test)
        classval_dots   = int(self.classdots * val)
        classtrain_dots = int(self.classdots * (1 - test - val))

        test_data  = []
        val_data   = []
        train_data = []

        for cl in range(self.classes):
            dz = dots[cl].copy()

            r.shuffle(dz)

            for t in range(classtest_dots):
                line = [cl] + dz.pop()
                test_data.append(line.copy())
            for v in range(classval_dots):
                line = [cl] + dz.pop()
                val_data.append(line.copy())
            for t in range(classtrain_dots):
                line = dz.pop()
                train_data.append(line.copy())

        r.shuffle(test_data)
        r.shuffle(val_data)
        r.shuffle(train_data)

        labels = ['cluster']

        for dim in range(self.dimensions):
            labels.append('X_' + str(dim))

        with open(join(directory, 'test.csv'), 'w') as test_file:
            w = csv.writer(test_file)

            w.writerow(labels)

            for line in range(len(test_data)):
                w.writerow(test_data[line])

        if val :
            with open(join(directory, 'val.csv'), 'w') as val_file:
                w = csv.writer(val_file)

                w.writerow(labels)

                for line in range(len(val_data)):
                    w.writerow(val_data[line])

        with open(join(directory, 'train.csv'), 'w') as train_file:
            w = csv.writer(train_file)

            labels.pop(0)
            w.writerow(labels)

            for line in range(len(train_data)):
                w.writerow(train_data[line])

        return


def generator(dimensions, classes, classdots, noise, intersections, borders):
    centers = gen_centers(dimensions=dimensions, classes=classes, borders=borders)
    centers = gen_noise(dimensions=dimensions, classes=classes, borders=borders, centers=centers, noise=noise, intersections=intersections)
    dots = gen_dots(dimensions=dimensions, classes=classes, classdots=classdots, centers=centers)
    data = gen_dict(dimensions=dimensions, classes=classes, classdots=classdots, noise=noise, intersections=intersections, borders=borders, centers=centers, dots=dots)

    return data


def gen_centers(
    dimensions, classes, borders
):
    from math import sqrt

    centers = {}

    min_rad = 2 * borders / ((sqrt(classes)) * 4)
    max_rad = 2 * borders / ((sqrt(classes)) * 1)

    centers[0] = {
        'center': np.array([0.0 for i in range(dimensions)]),
        'radius': r.uniform(min_rad, max_rad)
    }

    for cl in range(1, classes):
        potential_point = [r.uniform(-borders, borders) for i in range(dimensions)]
        check = False

        while not check:
            min_dist = 3 * borders

            check = True

            for cl_n, c in centers.items():
                dist = find_distance(potential_point, c['center'])
                if dist < (min_rad + c['radius']):
                    potential_point = [r.uniform(-borders, borders) for i in range(dimensions)]
                    check = False
                    continue
                else:
                    if dist < (min_dist + c['radius']):
                        min_dist = dist - c['radius']

            if min_dist == 3 * borders:
                potential_point = [r.uniform(-borders, borders) for i in range(dimensions)]
                check = False

        if min_dist > max_rad:
            radius = r.uniform(min_rad, max_rad)
        else:
            radius = min_dist
        centers[cl] = {
            'center': potential_point,
            'radius': radius
        }

    final_centers = {}

    rads = []
    inds = []

    for cl, c in centers.items():
        rads.append(c['radius'])
        inds.append(cl)

    rads, inds = zip(*sorted(zip(rads, inds)))

    for i in range(classes):
        final_centers[i] = centers[inds[i]]

    return final_centers


def gen_noise(dimensions, classes, centers, borders, noise=1, intersections=1):
    if not noise:
        return centers

    locked = []

    for intrsct in range(intersections):
        shortest = 3 * borders

        for cli in range(classes - 1):
            for clj in range(cli + 1, classes):
                if clj in locked:
                    continue
                dist = find_distance(centers[cli]['center'], centers[clj]['center'])

                if dist < shortest:
                    shortest = dist
                    min_cl = [cli, clj]

        cli, clj = min_cl

        vec      = find_vec(centers[cli]['center'], centers[clj]['center'])
        between  = shortest - (centers[cli]['radius'] + centers[clj]['radius'])

        if between > 0:
            offset = vec * between / shortest

            centers[clj]['center'] = (np.array(centers[clj]['center']) - offset).tolist()
            vec = find_vec(centers[cli]['center'], centers[clj]['center'])

        offset = vec * noise

        centers[clj]['center'] = (np.array(centers[clj]['center']) - offset).tolist()
        vec = find_vec(centers[cli]['center'], centers[clj]['center'])

        locked.append(clj)
        if cli not in locked:
            locked.append(cli)

        if len(locked) >= intersections:
            break

    return centers


def gen_dots(dimensions, classes, classdots, centers):

    dots = {}

    for cl in range(classes):
        dots[cl] = gen_sphere_dots(
            dimensions=dimensions, classdots=classdots, center=centers[cl]
        )

    return dots


def gen_sphere_dots(dimensions, classdots, center):
    import math as m

    dots = []

    radius = r.random() * center['radius']

    for p in range(classdots):
        coords = []
        phis   = []

        for d in range(dimensions):
            phi = r.uniform(0, 2 * m.pi)
            phis.append(phi)

        x = radius

        for d in range(dimensions):
            x *= m.sin(phis[d])

        coords.append(x)

        for dim in range(1, dimensions):
            x = radius * m.cos(phis[dim - 1])

            for d in range(dim, dimensions):
                x *= m.sin(phis[d])

            coords.append(x)

        coords = (np.array(coords) + np.array(center['center'])).tolist()

        dots.append(coords.copy())

    return dots


def gen_dict(dimensions, classes, classdots, noise, intersections, borders, centers, dots):
    generated = {
        'dimensions': dimensions,
        'classes': classes,
        'classdots': classdots,
        'noise': noise,
        'intersections': intersections,
        'borders': borders,
        'centers': centers,
        'dots': dots
    }

    return generated


def plot(dz, borders=10, show=True):
    import matplotlib.pyplot as plt

    dotx  = []
    doty  = []
    dotc  = []

    for cl in dz.keys():
        for dot in dz[cl]:
            dotx.append(dot[0])
            doty.append(dot[1])
            dotc.append(cl + 5)

    plt.scatter(dotx, doty, c=tuple(dotc), cmap=dcm(len(dotc) + 10, 'gist_stern'))

    if show:
        plt.show()

    return


def find_center(dots):
    center = np.array(dots[0])

    for p in range(1, len(dots)):
        center += np.array(dots[p])

    return list(center / (len(dots)))


def find_distance(p_1, p_2):
    distance = np.linalg.norm(np.array(p_2) - np.array(p_1))
    return distance


def find_vec(p_1, p_2):
    vec = np.array(p_2) - np.array(p_1)
    return vec


def dcm(N, base_cmap=None):
    import matplotlib.pyplot as plt
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def main():
    d = Generator(dimensions=2, classes=50, classdots=500, borders=10, noise=0, intersections=50)
    d.generate()
    d.csv('out')
    d.plot()

    return


if __name__ == '__main__':
    main()
