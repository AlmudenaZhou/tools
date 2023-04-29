from sklearn.preprocessing import StandardScaler

from tools.models.transformers.base import TargetedTransformer


class StandardScalerTransformer(TargetedTransformer):

    def _individual_fit(self, x, feature):
        return StandardScaler().fit(x[[feature]])
