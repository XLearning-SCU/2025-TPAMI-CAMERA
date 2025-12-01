def get_default_config(data_name):
    if data_name == 'Scene_15':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[59, 20],
            ),
            training=dict(
                cov=False,
                seed=1,
                num=15,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'NoisyMNIST':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[784, 784],
            ),
            training=dict(
                cov=True,
                seed=1,
                num=10,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'LandUse_21':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[59, 40],
            ),
            training=dict(
                cov=False,
                seed=0,
                num=21,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'Reuters':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[10, 10],
            ),
            training=dict(
                cov=False,
                seed=1,
                num=6,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'MNIST-USPS':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[784, 256],
            ),
            training=dict(
                cov=False,
                seed=1,
                num=10,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'cub_googlenet':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[1024, 300],
            ),
            training=dict(
                cov=False,
                seed=1,
                num=10,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'DHA':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[6144, 110],
            ),
            training=dict(
                cov=False,
                seed=0,
                num=23,
                dim=256,
                pretrain_epoch=50
            ),
        )

    elif data_name == 'UWA30':
        """The default configs."""
        return dict(
            Autoencoder=dict(
                in_dims=[6144, 110],
            ),
            training=dict(
                cov=False,
                seed=0,
                num=30,
                dim=256,
                pretrain_epoch=50
            ),
        )

    else:
        raise Exception('Undefined data_name')
