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
