from tools.misc import hash_object_with_sha256


class HashedFeature:

    """
    This class convert a unique string identifying the input feature into a hash_bucket.

    Use at categorical variables that:
        - has out-of-vocabulary problem/incomplete classes at training
        - high cardinality
        - cold start problem

    Steps:
        1. Uses a deterministic and portable hashing algorithm specified at hashing_method attribute
        2. Calculates the module of the hash number and the bucket_size

    .. note: It is recommended to use categories/n_buckets ~ 5 ratio.

    .. warning:
        The models experience a decrease in accuracy with these transformation, and it's important to be
        cautious about bucket collision to prevent further loss of accuracy.

    .. warning:
        When the distribution of observations by unique strings is highly skewed, we encounter data loss
        in the representation of each bucket.
    """

    def __init__(self, hash_bucket_size, hashing_method=hash_object_with_sha256):
        self.hash_bucket_size = hash_bucket_size
        self.encoding_method = hashing_method
