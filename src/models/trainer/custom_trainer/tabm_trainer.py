# src/models/trainer_custom/mptms/tabm_trainer.py
from pathlib import Path
from typing import Any
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tabm import TabM

from src.data.collator_class.collator_base.base_collator import BaseCollator
from src.models.trainer.base_trainer.base_trainer import BaseTrainer
from src.utils.metrics import compute_multitask_classification_metrics
from configs.model.model_configs import TabMConfig


class TabMTrainer(BaseTrainer):
    model_config: TabMConfig  # *중요: Type hint

    def __init__(
        self,
        *,
        work_dir: Path,
        data_collator: BaseCollator,
        model_config: TabMConfig,
        metadata: dict[str, Any] = None
    ):
        super().__init__(
            work_dir=work_dir,
            data_collator=data_collator,
            model_config=model_config,
            metadata=metadata
        )
        self.model: TabM | None = None
        self.meta: dict[str, Any] = {}

    # ---- 내부 유틸: numpy -> torch 텐서 ----
    def _numpy_to_tensors(self, X: np.ndarray, y: np.ndarray):
        X_num = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))  # (N, T)
        return X_num, y_t

    # ---- 내부 유틸: (X,y numpy)로부터 DataLoader 생성(추가 분할 없음) ----
    def _make_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool):
        X_num, y_t = self._numpy_to_tensors(X, y)
        ds = TensorDataset(X_num, y_t)
        def collate_fn(batch):
            xs, ys = zip(*batch)
            xs = torch.stack(xs)
            ys = torch.stack(ys)
            return xs, ys
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)

    def fit(self, train_dataset, valid_dataset):
        # 상위 공통 변환기: 사용자 Dataset -> (X,y) numpy
        X_tr, y_tr = self._dataset_to_numpy(train_dataset)  # y_tr: (N, T)
        X_va, y_va = self._dataset_to_numpy(valid_dataset)  # y_va: (M, T)

        train_loader = self._make_loader(X_tr, y_tr, self.model_config.batch_size, shuffle=True, drop_last=True)
        valid_loader = self._make_loader(X_va, y_va, self.model_config.batch_size, shuffle=False, drop_last=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        T = y_tr.shape[1]
        K = self.model_config.num_classes
        k_members = self.model_config.k
        D = X_tr.shape[1]

        # 단일 TabM: d_out = T*K
        self.model = TabM.make(
            n_num_features=D,
            cat_cardinalities=None,  # 기본선: 전부 수치 취급
            d_out=T * K,
            k=k_members
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )
        criterion_member = nn.CrossEntropyLoss(reduction="none")

        history: dict[str, Any] = {
            "train_loss": [],
            "valid_loss": [],
            "valid_acc_macro": []
        }

        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        best_epoch = -1

        for epoch in range(1, self.model_config.max_epochs + 1):
            # ---- train ----
            self.model.train()
            total_loss = 0.0
            n_items = 0

            for xb, yb in train_loader:  # xb: (B, D), yb: (B, T)
                xb = xb.to(device)
                yb = yb.to(device)

                y_pred = self.model(xb)  # (B, k, T*K)

                # 멤버별·스텝별 손실 평균
                losses_steps = []
                for t in range(T):
                    logits_t = y_pred[:, :, t*K:(t+1)*K]  # (B, k, K)
                    losses_members = []
                    for i in range(logits_t.shape[1]):
                        logits_i = logits_t[:, i, :]                  # (B, K)
                        target_t = yb[:, t]                           # (B,)
                        loss_i = criterion_member(logits_i, target_t) # (B,)
                        losses_members.append(loss_i.mean())
                    loss_t = torch.stack(losses_members).mean()        # 멤버 평균
                    losses_steps.append(loss_t)
                loss = torch.stack(losses_steps).mean()                # 스텝 평균

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                n_items += xb.size(0)

            train_loss = total_loss / n_items

            # ---- valid ----
            self.model.eval()
            val_loss_sum = 0.0
            val_items = 0

            # 정확도(매크로): 스텝별 정확도 평균
            correct_per_t = torch.zeros(T, dtype=torch.long)
            total_per_t = torch.zeros(T, dtype=torch.long)

            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    y_pred = self.model(xb)  # (B, k, T*K)

                    # 손실(검증도 동일 규칙)
                    v_losses_steps = []
                    for t in range(T):
                        logits_t = y_pred[:, :, t*K:(t+1)*K]  # (B, k, K)

                        v_losses_members = []
                        for i in range(logits_t.shape[1]):
                            logits_i = logits_t[:, i, :]
                            target_t = yb[:, t]
                            loss_i = criterion_member(logits_i, target_t).mean()
                            v_losses_members.append(loss_i)
                        v_loss_t = torch.stack(v_losses_members).mean()
                        v_losses_steps.append(v_loss_t)

                        # 추론: 확률 평균 → argmax
                        probs_t = torch.softmax(logits_t, dim=-1)  # (B, k, K)
                        probs_mean_t = probs_t.mean(dim=1)         # (B, K)
                        pred_t = probs_mean_t.argmax(dim=1)        # (B,)
                        correct_per_t[t] += (pred_t == yb[:, t]).sum().cpu()
                        total_per_t[t] += yb.size(0)

                    batch_vloss = torch.stack(v_losses_steps).mean()
                    val_loss_sum += batch_vloss.item() * xb.size(0)
                    val_items += xb.size(0)

            val_loss = val_loss_sum / val_items
            accs = []
            for t in range(T):
                if total_per_t[t] > 0:
                    accs.append(float(correct_per_t[t]) / float(total_per_t[t]))
            val_acc_macro = float(np.mean(accs)) if len(accs) > 0 else 0.0

            history["train_loss"].append(float(train_loss))
            history["valid_loss"].append(float(val_loss))
            history["valid_acc_macro"].append(float(val_acc_macro))

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        self.meta = {"T": int(T), "K": int(K), "k_members": int(k_members), "best_epoch": int(best_epoch)}
        return history

    def eval(self, test_dataset) -> dict[str, Any]:
        X_te, y_te = self._dataset_to_numpy(test_dataset)  # y_te: (N, T)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_num = torch.from_numpy(X_te.astype(np.float32)).to(device)
        T = int(self.meta["T"])
        K = int(self.meta["K"])

        with torch.no_grad():
            y_pred = self.model(X_num)                # (N, k, T*K)
            preds_each_t = []
            for t in range(T):
                logits_t = y_pred[:, :, t*K:(t+1)*K]  # (N, k, K)
                probs_t = torch.softmax(logits_t, dim=-1)
                probs_mean_t = probs_t.mean(dim=1)    # (N, K)
                pred_t = probs_mean_t.argmax(dim=1).detach().cpu().numpy()
                preds_each_t.append(pred_t)

        y_pred_np = np.stack(preds_each_t, axis=1).astype(np.int64)  # (N, T)
        return compute_multitask_classification_metrics(y_te, y_pred_np)

    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        torch.save(self.model.state_dict(), save_path / f"{self.model_config.save_model_name}.{self.model_config.model_ext}")
        return save_path

    def load(self, model_path: Path):
        with open(model_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta

        T = int(meta["T"])
        K = int(meta["K"])
        k_members = int(meta["k_members"])
        # D = int(meta["D"])  # fit()에서 저장된 입력 피처 수 사용

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TabM.make(
            n_num_features=240,
            cat_cardinalities=None,
            d_out=T * K,
            k=k_members
        ).to(device)

        state = torch.load(
            model_path / f"{self.model_config.save_model_name}.{self.model_config.model_ext}",
            map_location="cpu"
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(device)
        self.model.eval()
        return self.model
