"""A small PyTorch feed-forward classifier wrapped to behave like a
scikit-learn estimator (fit / predict / predict_proba), so it can sit in the
same evaluation harness as the four classic ML models.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch import nn

from . import config


class _Net(nn.Module):
    def __init__(self, n_in: int, n_classes: int, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-compatible wrapper around the PyTorch MLP above."""

    def __init__(
        self,
        hidden=(128, 64),
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=200,
        patience=15,
        val_fraction=0.15,
        batch_size=64,
        random_state=config.RANDOM_STATE,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        n_classes = len(self.classes_)

        # Chronological internal validation split (last slice = most recent
        # matches) for early stopping, consistent with the outer time-based
        # train/test split — no shuffling across time.
        n_val = max(1, int(len(X) * self.val_fraction))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y_enc[:-n_val], y_enc[-n_val:]

        class_counts = np.bincount(y_train, minlength=n_classes)
        class_weights = torch.tensor(
            len(y_train) / (n_classes * np.maximum(class_counts, 1)), dtype=torch.float32
        )

        self.model_ = _Net(X.shape[1], n_classes, self.hidden, self.dropout)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train).long()
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val).long()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        n = len(X_train_t)

        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                out = self.model_(X_train_t[idx])
                loss = criterion(out, y_train_t[idx])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(idx)
