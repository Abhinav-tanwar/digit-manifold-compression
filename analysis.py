import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from utils import OUT, save


def run():
    d = load_digits()
    a, test = train_test_split(
        np.arange(len(d.data)), test_size=0.2, stratify=d.target, random_state=24
    )
    train, val = train_test_split(
        a, test_size=0.25, stratify=d.target[a], random_state=24
    )
    scale = StandardScaler().fit(d.data[train])
    X = scale.transform(d.data[train])
    V = scale.transform(d.data[val])
    T = scale.transform(d.data[test])
    validation = {}
    models = {}
    for gamma in [0.001, 0.01, 0.1]:
        model = KernelPCA(
            n_components=2,
            kernel="rbf",
            gamma=gamma,
            fit_inverse_transform=True,
            alpha=1,
            eigen_solver="arpack",
            random_state=24,
        ).fit(X)
        rv = model.inverse_transform(model.transform(V))
        validation[str(gamma)] = float(np.mean((V - rv) ** 2))
        models[gamma] = model
    gamma = min(models, key=lambda g: validation[str(g)])
    kp = models[gamma]
    p = PCA(n_components=2, random_state=24).fit(X)
    a = p.inverse_transform(p.transform(T))
    b = kp.inverse_transform(kp.transform(T))
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    samples = [T[0], a[0], b[0]]
    for axis, image, title in zip(
        ax,
        samples,
        ["Held-out digit", "PCA reconstruction", "Kernel PCA reconstruction"],
    ):
        axis.imshow(
            scale.inverse_transform(image.reshape(1, -1)).reshape(8, 8), cmap="gray"
        )
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "reconstruction.png")
    plt.close(fig)
    return save(
        dict(
            train_rows=len(train),
            validation_rows=len(val),
            test_rows=len(test),
            selected_gamma=gamma,
            validation_standardized_mse=validation,
            pca_test_standardized_mse=float(np.mean((T - a) ** 2)),
            kernel_pca_test_standardized_mse=float(np.mean((T - b) ** 2)),
            scope="reconstruction in standardized feature space; gamma chosen without test data",
        )
    )


if __name__ == "__main__":
    run()
