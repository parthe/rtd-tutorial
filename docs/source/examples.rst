Examples
========


FashionMNIST
------------
Load the training data

.. code-block:: python

    import torch
    from torchvision import datasets
    
    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    
    X_all = train_dataset.data.flatten(1)/255.0
    Y = one_hot(train_dataset.targets.long())
    train_set_indices = np.random.choice(len(train_dataset), args.n, replace=False)
    
    X = X_all[train_set_indices,:]
    Y = Y[train_set_indices]
    
    centers_set_indices = np.random.choice(len(train_dataset), args.p, replace=False)
    Z = X_all[centers_set_indices,:]
    
    X_val = test_dataset.data.flatten(1)/255.0
    Y_val = one_hot(test_dataset.targets.long())

Create the EigenPro model and initialize the optimizer

.. code-block:: python

    from eigenpro.kernels import laplacian
    from eigenpro.models import KernelModel
    from eigenpro import trainer
