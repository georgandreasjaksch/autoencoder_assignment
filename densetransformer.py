from sklearn.base import TransformerMixin

class DenseTransformer(TransformerMixin):
    '''
        Creates a transformer class inheriting from TransformerMixin
        transforming a sparse to dense input (with todense()) as
        this is neccessary for example for PCA.
        Can be used in a pipeline in sklearn
    '''
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self