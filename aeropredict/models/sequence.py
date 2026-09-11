"""Sequence model: an LSTM over a window of normalised sensor readings.

Where the tabular models see hand-built rolling statistics, this one sees the raw
(normalised) window and has to learn the degradation trend itself. It is trained
with plain MSE on the capped RUL and early-stopped on a *unit-level* hold-out — a
random split of rows would put neighbouring cycles of the same engine on both
sides and make the validation loss meaningless.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import RUL_CAP


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMRUL:
    name = "lstm"

    def __init__(
        self,
        n_features: int,
        cap: int = RUL_CAP,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 40,
        patience: int = 6,
        seed: int = 0,
        device: str | None = None,
    ):
        self.n_features = n_features
        self.cap = cap
        self.hparams = dict(hidden=hidden, layers=layers, dropout=dropout)
        self.lr, self.batch_size, self.epochs, self.patience, self.seed = (
            lr, batch_size, epochs, patience, seed,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.net = _LSTMNet(n_features, hidden, layers, dropout).to(self.device)
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ train
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        val_fraction: float = 0.1,
        log=None,
    ) -> "LSTMRUL":
        """``groups`` is the unit id of each window, used for the hold-out split."""
        rng = np.random.default_rng(self.seed)
        units = np.unique(groups)
        val_units = rng.choice(units, size=max(1, int(len(units) * val_fraction)), replace=False)
        is_val = np.isin(groups, val_units)

        y_scaled = np.asarray(y, dtype=np.float32) / self.cap  # train on [0, 1]
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y_scaled)
        tr_idx = torch.as_tensor(np.flatnonzero(~is_val))
        va_idx = torch.as_tensor(np.flatnonzero(is_val))

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        loss_fn = nn.MSELoss()
        best, best_state, bad = float("inf"), None, 0
        gen = torch.Generator().manual_seed(self.seed)

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            perm = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
            total = 0.0
            for i in range(0, len(perm), self.batch_size):
                b = perm[i : i + self.batch_size]
                xb, yb = Xt[b].to(self.device), yt[b].to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            train_loss = total / len(perm)
            val_rmse = self._rmse(Xt[va_idx], yt[va_idx]) * self.cap
            sched.step(val_rmse)
            self.history.append({"epoch": epoch, "train_mse": train_loss, "val_rmse": val_rmse})
            if log:
                log(f"epoch {epoch:3d}  train_mse={train_loss:.4f}  val_rmse={val_rmse:.2f}")
            if val_rmse < best - 1e-3:
                best, bad = val_rmse, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _rmse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.net.eval()
        preds = self._forward_batched(X)
        return float(torch.sqrt(torch.mean((preds - y) ** 2)))

    @torch.no_grad()
    def _forward_batched(self, X: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, len(X), 2048):
            outs.append(self.net(X[i : i + 2048].to(self.device)).cpu())
        return torch.cat(outs) if outs else torch.empty(0)

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        pred = self._forward_batched(Xt).numpy() * self.cap
        return np.clip(pred, 0, self.cap)

    # ------------------------------------------------------------ persistence
    def state(self) -> dict:
        return {
            "n_features": self.n_features,
            "cap": self.cap,
            "hparams": self.hparams,
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "history": self.history,
        }

    @classmethod
    def from_state(cls, state: dict, device: str | None = None) -> "LSTMRUL":
        m = cls(state["n_features"], cap=state["cap"], device=device or "cpu", **state["hparams"])
        m.net.load_state_dict(state["state_dict"])
        m.net.to(m.device)
        m.history = state.get("history", [])
        return m
