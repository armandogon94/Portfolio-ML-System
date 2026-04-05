"""Demand forecasting LSTM trainer."""

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.device import get_device
from src.evaluation.timeseries_metrics import compute_timeseries_metrics
from src.features.timeseries_features import DemandDataset, prepare_timeseries
from src.models.lstm_forecaster import LSTMForecaster
from src.training.trainer import BaseTrainer, console


class DemandForecastTrainer(BaseTrainer):

    def __init__(self, use_wandb: bool = True):
        super().__init__("demand_forecasting", use_wandb=use_wandb)
        self.device = get_device()
        self.scalers = {}
        console.print(f"   Using device: [bold]{self.device}[/bold]")

    def load_data(self) -> pd.DataFrame:
        path = self.config["data"]["raw_data_path"]
        return pd.read_csv(path, parse_dates=["date"])

    def preprocess(self, df: pd.DataFrame) -> dict:
        window_size = self.config["features"]["window_size"]
        forecast_horizon = self.config["features"]["forecast_horizon"]
        test_days = self.config["data"]["test_days"]
        products = df["product_category"].unique()

        all_data = {}
        for product in products:
            pdata = prepare_timeseries(df, product, window_size, forecast_horizon, test_days)
            all_data[product] = pdata
            self.scalers[product] = pdata["scaler"]

        return {"product_data": all_data, "products": products}

    def train(self, data: dict) -> None:
        params = self.config["model"]["params"]

        # Train one model on all products combined
        X_train_all, y_train_all = [], []
        for product, pdata in data["product_data"].items():
            X_train_all.append(pdata["X_train"])
            y_train_all.append(pdata["y_train"])

        X_train = np.concatenate(X_train_all)
        y_train = np.concatenate(y_train_all)

        dataset = DemandDataset(X_train, y_train)
        loader = DataLoader(
            dataset,
            batch_size=params.get("batch_size", 64),
            shuffle=True,
        )

        forecast_horizon = self.config["features"]["forecast_horizon"]
        self.model = LSTMForecaster(
            input_size=params.get("input_size", 1),
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.2),
            forecast_horizon=forecast_horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.get("learning_rate", 0.001)
        )
        criterion = torch.nn.MSELoss()

        epochs = params.get("epochs", 100)
        patience = params.get("early_stopping_patience", 10)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.log_metric("train_loss", avg_loss, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                console.print(f"     Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            if patience_counter >= patience:
                console.print(f"     Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(best_state)

    def evaluate(self, data: dict) -> dict:
        self.model.eval()
        all_metrics = {}

        for product, pdata in data["product_data"].items():
            X_test = torch.FloatTensor(pdata["X_test"]).to(self.device)
            y_test = pdata["y_test"]

            with torch.no_grad():
                y_pred_scaled = self.model(X_test).cpu().numpy()

            # Inverse transform predictions and actuals
            scaler = pdata["scaler"]
            y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            metrics = compute_timeseries_metrics(y_true, y_pred, prefix=f"test_{product}")
            all_metrics.update(metrics)

        # Compute average across products
        mae_values = [v for k, v in all_metrics.items() if k.endswith("_mae")]
        rmse_values = [v for k, v in all_metrics.items() if k.endswith("_rmse")]
        all_metrics["test_avg_mae"] = float(np.mean(mae_values))
        all_metrics["test_avg_rmse"] = float(np.mean(rmse_values))

        return all_metrics

    def get_checkpoint_artifacts(self) -> dict:
        return {
            "lstm.pt": lambda path: torch.save(self.model.state_dict(), path),
            "scalers.pkl": lambda path: joblib.dump(self.scalers, path),
        }
