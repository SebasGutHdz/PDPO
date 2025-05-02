# https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py

import numpy as np
import math
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
from torchdyn.datasets import generate_moons


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200,dim=2):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        # data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        # data = data.astype("float32")
        # data = data * 2 + np.array([-1, -0.2])
        # return data

        x0,_ = generate_moons(batch_size,noise=0.2)
        x0 = 3*x0-1
        return x0.numpy()
    # elif data == "8gaussians":
    #     scale = 4.
    #     centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
    #                (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
    #                                                      1. / np.sqrt(2)), 
    #                                                      (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    #     centers = np.array(centers)*scale#[(scale * x, scale * y) for x, y in centers]
        
    #     cov_mat = np.eye(2)*.1
    #     noise = rng.multivariate_normal([0, 0], cov = cov_mat, size = batch_size)
        
    #     # idxs = rng.randint(0, 8, batch_size)
    #     elements_per_center = batch_size // 8

    #     idxs = np.array([i*np.ones(elements_per_center,dtype=np.int32) for i in range(8)]).flatten()

    #     if len(idxs) < batch_size:
    #         idxs = np.concatenate((idxs, 7*np.ones(batch_size-len(idxs),dtype=np.int32)))
        
    #     dataset = np.array([centers[idxs[i]] + noise[i] for i in range(batch_size)], dtype="float32")
        
        
    #     return dataset
    # elif data == "8gaussiansv2":
    #     scale = 8.
    #     centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
    #                (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
    #                                                      1. / np.sqrt(2)), 
    #                                                      (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    #     centers = np.array(centers)*scale#[(scale * x, scale * y) for x, y in centers]
        
    #     cov_mat = np.eye(2)*.1
    #     noise = rng.multivariate_normal([0, 0], cov = cov_mat, size = batch_size)
        
    #     # idxs = rng.randint(0, 8, batch_size)
    #     elements_per_center = batch_size // 8

    #     idxs = np.array([i*np.ones(elements_per_center,dtype=np.int32) for i in range(8)]).flatten()

    #     if len(idxs) < batch_size:
    #         idxs = np.concatenate((idxs, 7*np.ones(batch_size-len(idxs),dtype=np.int32)))
        
    #     dataset = np.array([centers[idxs[i]] + noise[i] for i in range(batch_size)], dtype="float32")
        
        
    #     return dataset
    elif data == "8gaussiansv3":
        scale = 16.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), 
                                                         (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = np.array(centers)*scale#[(scale * x, scale * y) for x, y in centers]
        
        cov_mat = np.eye(2)
        noise = rng.multivariate_normal([0, 0], cov = cov_mat, size = batch_size)
        
        # idxs = rng.randint(0, 8, batch_size)
        elements_per_center = batch_size // 8

        idxs = np.array([i*np.ones(elements_per_center,dtype=np.int32) for i in range(8)]).flatten()

        if len(idxs) < batch_size:
            idxs = np.concatenate((idxs, 7*np.ones(batch_size-len(idxs),dtype=np.int32)))
        
        dataset = np.array([centers[idxs[i]] + noise[i] for i in range(batch_size)], dtype="float32")
        
        
        return dataset
    elif data == "4gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        centers = np.array(centers)*scale#[(scale * x, scale * y) for x, y in centers]
        
        cov_mat = 1* np.eye(2)
        noise = rng.multivariate_normal([0, 0], cov = cov_mat, size = batch_size)
        
        # idxs = rng.randint(0, 8, batch_size)
        elements_per_center = batch_size // 4

        idxs = np.array([i*np.ones(elements_per_center,dtype=np.int32) for i in range(8)]).flatten()

        if len(idxs) < batch_size:
            idxs = np.concatenate((idxs, 3*np.ones(batch_size-len(idxs),dtype=np.int32)))
        
        dataset = np.array([centers[idxs[i]] + noise[i] for i in range(batch_size)], dtype="float32")
        

        return dataset

    elif data == "gaussian0":
        cov_matrix = np.array([[1,0],[0,1]])*0.5
        data = rng.multivariate_normal(mean=[-11, -1], cov=cov_matrix, size=batch_size)
        return data.astype("float32")

    elif data == "gaussian1":
        cov_matrix = np.array([[1,0],[0,1]])*0.5
        data = rng.multivariate_normal(mean=[11,1], cov=cov_matrix, size=batch_size)
        return data.astype("float32")
    elif data == "gaussian0_d":
        cov_matrix = np.eye(dim)*0.5
        mean = np.ones(dim)
        data = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=batch_size)
        return data.astype("float32")
    elif data == "gaussian1_d":
        cov_matrix = np.eye(dim)*0.5
        mean = -np.ones(dim)
        data = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=batch_size)
        return data.astype("float32")
    
    elif data == "gaussian0_s":
        cov_matrix = np.array([[1,0],[0,1]])*0.1
        data = rng.multivariate_normal(mean=[-2, -2], cov=cov_matrix, size=batch_size)
        return data.astype("float32")

    elif data == "gaussian1_s":
        cov_matrix = np.array([[1,0],[0,1]])*0.01
        data = rng.multivariate_normal(mean=[2,2], cov=cov_matrix, size=batch_size)
        return data.astype("float32")
    # elif data == "gaussian1_d":
    #     cov_matrix = np.array([[1,0],[0,1]])*0.5
    #     mean = -np.ones(dim)
    #     data = rng.multivariate_normal(mean=mean, cov=cov_matrix, size=batch_size)
    #     return data.astype("float32")
    
    elif data == "gauss0_opinion_2d":
        cov = np.array([[0.5, 0.0], [0.0, 0.25]])
        mean = np.array([0.0, 0.0])
        data = rng.multivariate_normal(mean=mean, cov=cov, size=batch_size)
        return data.astype("float32")
    elif data == "gauss1_opinion_2d":
        cov = 3*np.array([[1.0, 0.0], [0.0, 1.0]])
        mean = np.array([0.0, 0.0])
        data = rng.multivariate_normal(mean=mean, cov=cov, size=batch_size)
        return data.astype("float32")
    elif data == "gauss0_opinion_1000d":
        cov = np.eye(1000)*0.25
        cov[0,0] = 4
        mean = np.zeros(1000)
        data = rng.multivariate_normal(mean=mean, cov=cov, size=batch_size)
        return data.astype("float32")
    elif data == "gauss1_opinion_1000d":
        cov = np.eye(1000)*3
        mean = np.zeros(1000)
        data = rng.multivariate_normal(mean=mean, cov=cov, size=batch_size)
        return data.astype("float32")
    
    elif data == "half_std_gaussian":
        data = rng.randn(batch_size, 2)*.5
        return data.astype("float32")
    elif data == "std_gaussian":
        data = rng.randn(batch_size, 2)
        return data.astype("float32")

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    
    
    elif data == "checkerboard":

        grid_size = 4
        # Calculate points per square to distribute batch_size evenly
        total_squares = (grid_size * grid_size) // 2  # Only half squares are filled
        points_per_square = batch_size // total_squares
        
        # Initialize arrays
        points = []
        square_indices = []
        
        # Generate points square by square
        for i in range(grid_size):
            for j in range(grid_size):
                # Check if current square should be filled (checkerboard pattern)
                if (i + j) % 2 == 0:
                    # Generate points for this square
                    x1 = np.random.uniform(i*2-4, (i+1)*2-4, points_per_square)
                    x2 = np.random.uniform(j*2-4, (j+1)*2-4, points_per_square)
                    
                    # Store points and their square index
                    square_points = np.stack([x1, x2], axis=1)
                    points.append(square_points)
                    square_indices.extend([i * grid_size + j] * points_per_square)
        
        # Combine all points and convert to numpy arrays
        points = np.concatenate(points, axis=0)
        square_indices = np.array(square_indices)
        
        # Shuffle within each square to avoid artificial patterns
        for idx in np.unique(square_indices):
            mask = square_indices == idx
            perm = np.random.permutation(np.sum(mask))
            points[mask] = points[mask][perm]
        
        return points#, square_indices

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)

    
    else:
        return inf_train_gen("8gaussians", rng, batch_size)