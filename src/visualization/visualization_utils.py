import matplotlib.pyplot as plt


class Visualization:
    @staticmethod
    def imshow(data, fig_num=None, xlabel="x", ylabel="y", title=None, colorbar=True, show=False):
        if fig_num is not None:
            plt.figure(fig_num)
        plt.imshow(data, origin='lower')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if colorbar:
            plt.colorbar()
        plt.title(title)
        if show:
            plt.show()

    @staticmethod
    def plot_two_data(x1, y1, x2, y2, label1="Numerical", label2="Analytical", xlabel="x", ylabel="Solution"):
        plt.plot(x1, y1, '.', label=label1)
        plt.plot(x2, y2, label=label2)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
