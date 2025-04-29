import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

class DecisionTree__Classifier:
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict_sample(self.tree, sample) for sample in X]

    def _gini(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            gini -= p ** 2
        return gini

    def _entropy(self, y):
        classes = np.unique(y)
        entropy = 0.0
        for cls in classes:
            p = len(y[y == cls]) / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_impurity = float('inf')
        impurity_fn = self._gini if self.criterion == 'gini' else self._entropy

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                impurity = (len(y_left) * impurity_fn(y_left) + len(y_right) * impurity_fn(y_right)) / len(y)
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return {
                'type': 'leaf',
                'class': np.bincount(y).argmax(),
                'samples': len(y),
                'value': np.bincount(y, minlength=3),#=len(np.unique(y))
                'impurity': 0.0
            }
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {
                'type': 'leaf',
                'class': np.bincount(y).argmax(),
                'samples': len(y),
                'value': np.bincount(y, minlength=3),#len(np.unique(y))
                'impurity': 0.0
            }

        X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)
        impurity_fn = self._gini if self.criterion == 'gini' else self._entropy
        return {
            'type': 'node',
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
            'samples': len(y),
            'value': np.bincount(y, minlength=len(np.unique(y))),
            'impurity': impurity_fn(y)
        }

    def _predict_sample(self, node, sample):
        if node['type'] == 'leaf':
            return node['class']
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(node['left'], sample)
        else:
            return self._predict_sample(node['right'], sample)

    def export__graphviz(self, feature_names, class_names):
        def node_to_str(node):
            if node['type'] == 'leaf':
                class_name = class_names[node['class']]
                return f"leaf node: class={class_name}\\nsamples = {node['samples']}\\nvalue = {node['value'].tolist()}\\nclass = {class_name}"
            else:
                class_name = class_names[node['value'].argmax()]
                impurity_name = 'entropy' if self.criterion == 'entropy' else 'gini'
                return f"{feature_names[node['feature']]} <= {node['threshold']}\\n{impurity_name} = {node['impurity']:.3f}\\nsamples = {node['samples']}\\nvalue = {node['value'].tolist()}\\nclass = {class_name}"

        def recurse(node, dot_data, parent=None):
            node_str = node_to_str(node)
            node_id = id(node)
            if parent is not None:
                dot_data.append(f'    {parent} -> {node_id};')
            if node['type'] == 'leaf':
                dot_data.append(f'    {node_id} [label="{node_str}", shape="box", style="filled", color="#c5e3ff"];')
            else:
                dot_data.append(f'    {node_id} [label="{node_str}", style="filled", color="#b8ffa9"];')
                recurse(node['left'], dot_data, node_id)
                recurse(node['right'], dot_data, node_id)

        dot_data = ['digraph Tree {', '    node [shape=box, style="filled", color="lightgrey"];', '    edge [fontname="helvetica"];']
        recurse(self.tree, dot_data)
        dot_data.append('}')
        return '\n'.join(dot_data)
    