import numpy as np
import matplotlib.pyplot as plt

from redundant_op import test_redundancy


if __name__ == '__main__':
    categories = []
    results = {
        'matched': [],
        'stride exception': [],
        'out of index exception': [],
        'unknown exception': [],
        'total operation': [],
    }

    for ifm_siz in range(4, 65, 4):
        tr = test_redundancy(ifm_siz, ifm_siz, FW=3, FH=3, S=1)
        categories.append(ifm_siz)
        total_op = 0

        for key, val in tr.items():
            results[key].append(tr[key])
            total_op += tr[key]

        results['total operation'].append(total_op)

    categories = np.array(categories)
    for key in results.keys():
        results[key] = np.array(results[key]) / np.array(results['total operation']) * 100

    print(results)
    print(categories)

    results['exception'] = results['stride exception'] + results['out of index exception'] + results['unknown exception']

    del results['stride exception']
    del results['out of index exception']
    del results['unknown exception']
    del results['total operation']

    width_max = 0.8
    width = width_max / len(results.keys())

    x_axis = np.arange(len(categories))
    for idx, (key, val) in enumerate(results.items()):
        xval = x_axis + ((idx - (len(results.keys()) / 2) + 0.5) * width)
        plt.bar(xval, val, width=width, label=key)
        for i, j in zip(xval, val):
            plt.annotate(f"{j:.2f}", xy=(i, j + 1), ha='center', size=6)
    plt.xticks(x_axis, list(map(lambda x: f"{x}x{x}", categories)), rotation=0, ha='center')

    plt.title("Distance pattern by the size of IFM")
    plt.xlabel('shape of input feature map')
    plt.ylabel('ratio of each pattern [%]')
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.show()