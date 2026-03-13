import numpy as np

class TWSM:

    def fit(self, X, y):
        self.autism_centroid = np.mean(X[y == 1], axis=0)
        self.normal_centroid = np.mean(X[y == 0], axis=0)

    def similarity(self, a, b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def predict(self, X):

        results=[]

        for x in X:

            s_aut = self.similarity(x, self.autism_centroid)
            s_norm = self.similarity(x, self.normal_centroid)

            score = s_aut - s_norm

            if score < 0.3:
                results.append("Mild")

            elif score < 0.6:
                results.append("Moderate")

            else:
                results.append("Severe")

        return results
