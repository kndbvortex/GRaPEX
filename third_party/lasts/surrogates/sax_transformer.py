from lasts.surrogates.sax_tree import SaxTree
from aeon.classification.dictionary_based import MrSEQLClassifier
from aeon.classification.dictionary_based._mrseql import (
    _from_numpy3d_to_nested_dataframe,
)


class SaxTransformer(SaxTree):
    def __init__(
        self,
        labels=None,
        random_state=None,
        create_plotting_dictionaries=True,
        custom_config=None,
    ):
        super().__init__(
            labels=labels,
            random_state=random_state,
            create_plotting_dictionaries=create_plotting_dictionaries,
            custom_config=custom_config,
        )

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        X_aeon = X.transpose(0, 2, 1)  # (n_samples, n_timepoints, n_channels) -> (n_cases, n_channels, n_timepoints)
        seql_model = MrSEQLClassifier(
            seql_mode="clf", symrep="sax", custom_config=self.custom_config
        )
        seql_model.fit(X_aeon, y)
        X_nested = _from_numpy3d_to_nested_dataframe(X_aeon)
        mr_seqs = seql_model.clf_._transform_time_series(X_nested)
        X_transformed = seql_model.clf_._to_feature_space(mr_seqs)
        self.X_transformed_ = X_transformed
        self.seql_model_ = seql_model.clf_  # underlying mrseql classifier
        if self.create_plotting_dictionaries:
            self._create_dictionaries()
        return self

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def explain(self, x, **kwargs):
        pass
