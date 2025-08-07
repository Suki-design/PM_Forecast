import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from metrics import rmse, mae, r2

def make_loader(X, y, batch_size=16, shuffle=False, seed=42):
    ds = TensorDataset(X, y)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=g)

def train_one_city_cpu(model, x_train, y_train, x_valid, y_valid,
                       lr=2e-3, batch_size=16, max_epochs=60, patience=8, seed=42):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    tr_loader = make_loader(x_train, y_train, batch_size, True, seed)
    va_loader = make_loader(x_valid, y_valid, batch_size, False, seed)

    best_rmse, bad, best_state = np.inf, 0, None
    for _ in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # compute validation rmse for early stopping
        model.eval()
        with torch.no_grad():
            val_preds = torch.cat([model(xb) for xb, _ in va_loader]).numpy()
            val_true  = torch.cat([yb      for _,  yb in va_loader]).numpy()
        v_rmse = rmse(val_preds, val_true)
        if v_rmse < best_rmse - 1e-4:
            best_rmse, bad = v_rmse, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad == patience: break

    # restore best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = torch.cat([model(xb) for xb, _ in va_loader]).numpy()
        val_true  = torch.cat([yb      for _,  yb in va_loader]).numpy()

    return {
        "valid_rmse": rmse(val_preds, val_true),
        "valid_mae" : mae (val_preds, val_true),
        "valid_r2"  : r2  (val_preds, val_true),
        "model"     : model
    }

def retrain_on_train_valid_cpu(model, x_train, y_train, x_valid, y_valid,
                               x_test, y_test, lr=2e-3, batch_size=16, epochs=10, seed=42):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    X = torch.cat([x_train, x_valid]); y = torch.cat([y_train, y_valid])
    loader = make_loader(X, y, batch_size, True, seed)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(x_test).numpy()
        test_true  = y_test.numpy()
    return {
        "test_rmse": rmse(test_preds, test_true),
        "test_mae" : mae (test_preds, test_true),
        "test_r2"  : r2  (test_preds, test_true)
    }
