from tflearn.data_preprocessing import DataPreprocessing
import numpy as np
import random

class ImagePreprocessing3d(DataPreprocessing):
    """ Image Preprocessing.
    Base class for applying real-time image related pre-processing.
    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined pre-processing methods will be applied at both
    training and testing time. Note that ImageAugmentation is similar to
    ImagePreprocessing, but only applies at training time.
    """

    def __init__(self):
        super(ImagePreprocessing3d, self).__init__()
        self.global_mean_pc = False
        self.global_std_pc = False

    # -----------------------
    #  Preprocessing Methods
    # -----------------------

    def add_image_normalization(self):
        """ add_image_normalization.
        Normalize a picture pixel to 0-1 float (instead of 0-255 int).
        Returns:
            Nothing.
        """
        self.methods.append(self._normalize_image)
        self.args.append(None)

    def add_crop_center(self, shape):
        """ add_crop_center.
        Crop the center of an image.
        Arguments:
            shape: `tuple` of `int`. The croping shape (height, width).
        Returns:
            Nothing.
        """
        self.methods.append(self._crop_center)
        self.args.append([shape])

    def resize(self, height, width):
        raise NotImplementedError

    def blur(self):
        raise NotImplementedError

    # -----------------------
    #  Preprocessing Methods
    # -----------------------

    def _normalize_image(self, batch):
        return np.array(batch) / 255.

    def _crop_center(self, batch, shape):
        oshape = np.shape(batch[0])
        nd = int((oshape[0] - shape[0]) * 0.5)
        nh = int((oshape[0] - shape[1]) * 0.5)
        nw = int((oshape[1] - shape[2]) * 0.5)
        new_batch = []
        for i in range(len(batch)):
            new_batch.append(batch[i][nd: nd + shape[0], nh: nh + shape[1], nw: nw + shape[2]])
        return new_batch

    # ----------------------------------------------
    #  Preprocessing Methods (Overwritten from Base)
    # ----------------------------------------------

    def add_samplewise_zero_center(self, per_channel=False):
        """ add_samplewise_zero_center.
        Zero center each sample by subtracting it by its mean.
        Arguments:
            per_channel: `bool`. If True, apply per channel mean.
        Returns:
            Nothing.
        """
        self.methods.append(self._samplewise_zero_center)
        self.args.append([per_channel])

    def add_samplewise_stdnorm(self, per_channel=False):
        """ add_samplewise_stdnorm.
        Scale each sample with its standard deviation.
        Arguments:
            per_channel: `bool`. If True, apply per channel std.
        Returns:
            Nothing.
        """
        self.methods.append(self._samplewise_stdnorm)
        self.args.append([per_channel])

    def add_featurewise_zero_center(self, mean=None, per_channel=False):
        """ add_samplewise_zero_center.
        Zero center every sample with specified mean. If not specified,
        the mean is evaluated over all samples.
        Arguments:
            mean: `float` (optional). Provides a custom mean. If none
                provided, it will be automatically caluclated based on
                the training dataset. Default: None.
            per_channel: `bool`. If True, compute mean per color channel.
        Returns:
            Nothing.
        """
        self.global_mean.is_required = True
        self.global_mean.value = mean
        if per_channel:
            self.global_mean_pc = True
        self.methods.append(self._featurewise_zero_center)
        self.args.append(None)

    def add_featurewise_stdnorm(self, std=None, per_channel=False):
        """ add_featurewise_stdnorm.
        Scale each sample by the specified standard deviation. If no std
        specified, std is evaluated over all samples data.
        Arguments:
            std: `float` (optional). Provides a custom standard derivation.
                If none provided, it will be automatically caluclated based on
                the training dataset. Default: None.
            per_channel: `bool`. If True, compute std per color channel.
        Returns:
            Nothing.
        """
        self.global_std.is_required = True
        self.global_std.value = std
        if per_channel:
            self.global_std_pc = True
        self.methods.append(self._featurewise_stdnorm)
        self.args.append(None)

    # --------------------------------------------------
    #  Preprocessing Calculation (Overwritten from Base)
    # --------------------------------------------------

    def _samplewise_zero_center(self, batch, per_channel=False):
        for i in range(len(batch)):
            if not per_channel:
                batch[i] -= np.mean(batch[i])
            else:
                batch[i] -= np.mean(batch[i], axis=(0, 1, 2), keepdims=True)
        return batch

    def _samplewise_stdnorm(self, batch, per_channel=False):
        for i in range(len(batch)):
            if not per_channel:
                batch[i] /= (np.std(batch[i]) + _EPSILON)
            else:
                batch[i] /= (np.std(batch[i], axis=(0, 1, 2),
                                    keepdims=True) + _EPSILON)
        return batch

    # --------------------------------------------------------------
    #  Calulation with Persistent Parameters (Overwritten from Base)
    # --------------------------------------------------------------

    def _compute_global_mean(self, dataset, session, limit=None):
        """ Compute mean of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements. """
        _dataset = dataset
        mean = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray) and not self.global_mean_pc:
            mean = np.mean(_dataset)
        else:
            # Iterate in case of non numpy data
            for i in range(len(dataset)):
                if not self.global_mean_pc:
                    mean += np.mean(dataset[i]) / len(dataset)
                else:
                    mean += (np.mean(dataset[i], axis=(0, 1, 2),
                             keepdims=True) / len(dataset))[0][0][0]
        self.global_mean.assign(mean, session)
        return mean

    def _compute_global_std(self, dataset, session, limit=None):
        """ Compute std of a dataset. A limit can be specified for faster
        computation, considering only 'limit' first elements. """
        _dataset = dataset
        std = 0.
        if isinstance(limit, int):
            _dataset = _dataset[:limit]
        if isinstance(_dataset, np.ndarray) and not self.global_std_pc:
            std = np.std(_dataset)
        else:
            for i in range(len(dataset)):
                if not self.global_std_pc:
                    std += np.std(dataset[i]) / len(dataset)
                else:
                    std += (np.std(dataset[i], axis=(0, 1, 2),
                             keepdims=True) / len(dataset))[0][0][0]
        self.global_std.assign(std, session)
        return std