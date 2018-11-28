import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def generate_points():
    x = []
    y = []
    for i in range(1, 301):
        x.append(numpy.random.uniform(-1, 1))
        y.append(numpy.random.uniform(-1, 1))
    return x, y


def assign_class(x, y, w0, w1, w2):
    classes = []
    for xi, yi in zip(x, y):
        classes.append(numpy.sign(w0 + w1*xi + w2*yi))
    return classes, x, y


def plot_points(classes, x, y):
    h = .02
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])

    #clf = GaussianNB()
    #clf = LogisticRegression()
    clf = KNeighborsClassifier()

    X = numpy.column_stack((x, y))
    clf.fit(X, classes)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    colors = []
    for value in classes:
        if value == 1:
            colors.append('red')
        else:
            colors.append('blue')
    for i, value in enumerate(x):
        plt.scatter(x[i], y[i], color=colors[i])

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.show()


if __name__ == '__main__':
    x, y = generate_points()
    classes, x, y = assign_class(x, y, 0.5, 4, 1)
    plot_points(classes, x, y)