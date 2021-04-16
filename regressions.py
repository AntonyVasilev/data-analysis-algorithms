import numpy as np


class linear_regression:

    def __init__(self, eta=0.9, max_iter=1e4, min_weight_dist=1e-8):
        self.eta = eta
        self.max_iter = max_iter
        self.min_weight_dist = min_weight_dist

    def _mserror(self, X, y_real):
        # расчёт среднеквадратичной ошибки
        y = X.dot(self.w.T) + self.w0
        return np.sum((y - y_real)**2) / y_real.shape[0]

    def _mserror_grad(self, X, y_real):
        # расчёт градиента ошибки.
        # 2*delta.T.dot(X)/y_real.shape[0] - градиент по коэффициентам при факторах
        # np.sum(2*delta)/y_real.shape[0] - производная(градиент) при нулевом коэффициенте
        delta = (X.dot(self.w.T) + self.w0 - y_real)
        return 2*delta.T.dot(X) / y_real.shape[0], np.sum(2*delta) / y_real.shape[0]

    def _optimize(self, X, Y):
        # оптимизация коэффициентов
        iter_num = 0
        weight_dist = np.inf
        self.w = np.zeros((1, X.shape[1]))
        self.w0 = 0
        while weight_dist > self.min_weight_dist and iter_num < self.max_iter:
            gr_w, gr_w0 = self._mserror_grad(X, Y)
            if iter_num == 0:
                # Чтобы eta адаптировалась к порядку градиента, делим на l2 норму градиента в нуле
                eta = self.eta/np.sqrt(np.linalg.norm(gr_w)**2 + (gr_w0)**2)
            new_w = self.w - eta * gr_w
            new_w0 = self.w0 - eta * gr_w0
            weight_dist = np.sqrt(np.linalg.norm(new_w - self.w)**2 + (new_w0 - self.w0)**2)
            iter_num += 1
            self.w = new_w
            self.w0 = new_w0

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        self._optimize(X, Y)

    def predict(self, X):
        return (X.dot(self.w.T)+self.w0).flatten()

    def test(self, X, Y):
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        return self._mserror(X, Y)


class polynomial_regression(linear_regression):

    def __init__(self, max_power, *args, **kwargs):
        self.max_power = max_power
        super().__init__(*args, **kwargs)

    @staticmethod
    def generate_features(x, max_power):
        x = x[:, np.newaxis]
        return np.concatenate([x ** i for i in range(1, max_power + 1)], axis=1)

    def fit(self, x, y):
        super().fit(self.generate_features(x, self.max_power), y)

    def predict(self, x):
        return super().predict(self.generate_features(x, self.max_power)).flatten()

    def test(self, x, y):
        return super().test(self.generate_features(x, self.max_power), y)


class logistic_regression:

    def __init__(self, n_iterations=1000, eta=0.05):
        self.n_iterations = n_iterations
        self.eta = eta

    def _log_grad(self, X, target):
        m = X.shape[0]
        y = (2*target - 1)
        score = np.dot(X, self.w.T).flatten() + self.w0
        Z = -y/(m * (1 + np.exp(y * score)))
        grad = Z[np.newaxis, :].dot(X)
        return grad/m, np.sum(Z)/m

    def _optimize(self, X, target):
        for i in range(self.n_iterations):        
            grad_w, grad_w0 = self._log_grad(X, target)
            self.w = self.w - self.eta * grad_w
            self.w0 = self.w0 - self.eta * grad_w0

    def fit(self, X, target):
        self.w = np.zeros((1, X.shape[1]))
        self.w0 = 0
        self._optimize(X, target)

    def predict_proba(self, X):  
        # Рассчёт вероятности
        score = X.dot(self.w.T).flatten() + self.w0
        return 1 / (1 + np.exp(-score))

    def predict(self, X, thr=0.5):
        proba = self.predict_proba(X)
        y_predicted = np.zeros(proba.shape, dtype=bool) 
        y_predicted[proba > thr] = 1
        y_predicted[proba <= thr] = 0
        return y_predicted
