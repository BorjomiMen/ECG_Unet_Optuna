import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

data_path = 'your path to dataset'

fs = 500
record_duration = 10
expected_samples = fs * record_duration

all_records = [f"{i}" for i in range(1, 201)]
train_records, test_val_records = train_test_split(all_records, test_size=40, random_state=42)
val_records, test_records = train_test_split(test_val_records, test_size=20, random_state=42)

classes_config = {
    1: {'before': int(0.1 * fs), 'after': int(0.1 * fs)},
    2: {'before': int(0.08 * fs), 'after': int(0.08 * fs)},
    3: {'before': int(0.2 * fs), 'after': int(0.2 * fs)}
}

def annotations_to_tensor(annotation, signal_length, fs, classes_config):
    tensor = torch.zeros(signal_length, dtype=torch.long)
    symbol_to_class = {'p': 1, 'N': 2, 't': 3}
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in symbol_to_class:
            continue
        class_id = symbol_to_class[symbol]
        config = classes_config[class_id]
        start = max(0, sample - config['before'])
        end = min(signal_length, sample + config['after'])
        mask = (tensor[start:end] < class_id) | (tensor[start:end] == 0)
        tensor[start:end][mask] = class_id
    return tensor

class ECGDataset(Dataset):
    def __init__(self, records, channels, data_path, fs, expected_samples, classes_config):
        self.signals = []
        self.masks = []
        self.fs = fs
        self.expected_samples = expected_samples
        self.classes_config = classes_config
        for record_name in tqdm(records, desc="Processing records"):
            record_path = os.path.join(data_path, record_name)
            if not os.path.exists(f"{record_path}.hea"):
                continue
            try:
                record = wfdb.rdrecord(record_path)
            except:
                continue
            for channel in channels:
                if channel not in record.sig_name:
                    continue
                ch_idx = record.sig_name.index(channel)
                signal = record.p_signal[:, ch_idx].flatten()
                if len(signal) != expected_samples:
                    continue
                try:
                    annotation = wfdb.rdann(record_path, channel)
                except:
                    continue
                mask = annotations_to_tensor(
                    annotation=annotation,
                    signal_length=len(signal),
                    fs=self.fs,
                    classes_config=classes_config
                )
                self.signals.append(signal)
                self.masks.append(mask)
        if not self.signals:
            raise ValueError("Нет данных после обработки")
        self.signals = np.array(self.signals, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=np.int64)
    def __len__(self):
        return len(self.signals)
    def __getitem__(self, idx):
        signal = self.signals[idx]
        mask = self.masks[idx]
        std = np.std(signal)
        if std == 0:
            signal = np.zeros_like(signal)
        else:
            signal = (signal - np.mean(signal)) / std
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.long)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, dropout_rate=0.2):
        super().__init__()
        self.inc = self._conv_block(in_channels, 64, dropout_rate)
        self.down1 = self._down_block(64, 128, dropout_rate)
        self.down2 = self._down_block(128, 256, dropout_rate)
        self.down3 = self._down_block(256, 512, dropout_rate)
        self.up1 = self._up_block(512, 256)
        self.up2 = self._up_block(256, 128)
        self.up3 = self._up_block(128, 64)
        self.outc = nn.Conv1d(64, num_classes, kernel_size=1)
    def _conv_block(self, in_ch, out_ch, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_rate),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_rate)
        )
    def _down_block(self, in_ch, out_ch, dropout_rate):
        return nn.Sequential(
            nn.MaxPool1d(2),
            self._conv_block(in_ch, out_ch, dropout_rate)
        )
    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        input_size = x.size(-1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        x = self.outc(x)
        if x.size(-1) != input_size:
            pad = input_size - x.size(-1)
            x = nn.functional.pad(x, (0, pad), mode='reflect')
        return x

def calculate_metrics(predictions, masks):
    """Вычисление F1 для каждой волны отдельно"""
    metrics = {1: [], 2: [], 3: []}
    for class_id in [1, 2, 3]:
        tp = ((predictions == class_id) & (masks == class_id)).sum().float()
        fp = ((predictions == class_id) & (masks != class_id)).sum().float()
        fn = ((predictions != class_id) & (masks == class_id)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[class_id].append(f1.item())
    return metrics

def plot_metrics_history(history, channel):
    epochs = range(1, len(history['train'][1]) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train'][1], label='Train P')
    plt.plot(epochs, history['val'][1], label='Val P')
    plt.title(f'{channel.upper()} - P Wave')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train'][2], label='Train QRS')
    plt.plot(epochs, history['val'][2], label='Val QRS')
    plt.title(f'{channel.upper()} - QRS Complex')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train'][3], label='Train T')
    plt.plot(epochs, history['val'][3], label='Val T')
    plt.title(f'{channel.upper()} - T Wave')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.tight_layout()
    plt.show()

def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16])
    }
    val_f1 = train_optimized('ii', params)
    return val_f1[1]

def train_optimized(channel, params):
    train_dataset = ECGDataset(
        records=train_records,
        channels=[channel],
        data_path=data_path,
        fs=fs,
        expected_samples=expected_samples,
        classes_config=classes_config
    )
    val_dataset = ECGDataset(
        records=val_records,
        channels=[channel],
        data_path=data_path,
        fs=fs,
        expected_samples=expected_samples,
        classes_config=classes_config
    )
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D(num_classes=4, dropout_rate=params['dropout_rate']).to(device)
    class_counts = np.bincount(np.concatenate(train_dataset.masks))
    weights = torch.tensor(class_counts.sum() / (class_counts * len(class_counts)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1_p = 0
    patience = 10
    counter = 0
    for epoch in range(50):
        model.train()
        for signals, masks in train_loader:
            signals, masks = signals.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(signals)
                loss = criterion(outputs, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val_f1 = {1: [], 2: [], 3: []}
        with torch.no_grad():
            for signals, masks in val_loader:
                signals, masks = signals.to(device), masks.to(device)
                outputs = model(signals)
                probs = torch.softmax(outputs, 1)
                _, preds = torch.max(probs, dim=1)
                f1_scores = calculate_metrics(preds, masks)
                for class_id in [1, 2, 3]:
                    val_f1[class_id].extend(f1_scores[class_id])
        avg_val_f1 = {k: np.mean(v) for k, v in val_f1.items()}
        if avg_val_f1[1] > best_val_f1_p:
            best_val_f1_p = avg_val_f1[1]
            counter = 0
            torch.save(model.state_dict(), f'best_unet_optuna_{channel}.pth')
        else:
            counter += 1
            if counter >= patience:
                break
    return avg_val_f1

def train(channel, use_best_params=False, best_params=None):
    try:
        train_dataset = ECGDataset(
            records=train_records,
            channels=[channel],
            data_path=data_path,
            fs=fs,
            expected_samples=expected_samples,
            classes_config=classes_config
        )
        val_dataset = ECGDataset(
            records=val_records,
            channels=[channel],
            data_path=data_path,
            fs=fs,
            expected_samples=expected_samples,
            classes_config=classes_config
        )
        test_dataset = ECGDataset(
            records=test_records,
            channels=[channel],
            data_path=data_path,
            fs=fs,
            expected_samples=expected_samples,
            classes_config=classes_config
        )
    except ValueError as e:
        print(f"Пропуск канала {channel}: {str(e)}")
        return
    batch_size = best_params['batch_size'] if use_best_params else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dropout_rate = best_params['dropout_rate'] if use_best_params else 0.2
    model = UNet1D(num_classes=4, dropout_rate=dropout_rate).to(device)
    class_counts = np.bincount(np.concatenate(train_dataset.masks))
    weights = torch.tensor(class_counts.sum() / (class_counts * len(class_counts)), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    lr = best_params['lr'] if use_best_params else 0.0001
    weight_decay = best_params['weight_decay'] if use_best_params else 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1_p = 0
    patience = 10
    counter = 0
    channel_history = {
        'train': {1: [], 2: [], 3: []},
        'val': {1: [], 2: [], 3: []}
    }
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        train_f1_p = 0.0
        train_f1_qrs = 0.0
        train_f1_t = 0.0
        for signals, masks in tqdm(train_loader, desc=f'[Epoch {epoch+1}] Train', leave=False):
            signals, masks = signals.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(signals)
                loss = criterion(outputs, masks)
                probs = torch.softmax(outputs, 1)
                _, preds = torch.max(probs, 1)
                f1_scores = calculate_metrics(preds, masks)
                train_f1_p += f1_scores[1][0]
                train_f1_qrs += f1_scores[2][0]
                train_f1_t += f1_scores[3][0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_train_f1_p = train_f1_p / len(train_loader)
        avg_train_f1_qrs = train_f1_qrs / len(train_loader)
        avg_train_f1_t = train_f1_t / len(train_loader)
        model.eval()
        val_loss = 0.0
        val_f1_p = 0.0
        val_f1_qrs = 0.0
        val_f1_t = 0.0
        with torch.no_grad():
            for signals, masks in tqdm(val_loader, desc=f'[Epoch {epoch+1}] Val', leave=False):
                signals, masks = signals.to(device), masks.to(device)
                outputs = model(signals)
                probs = torch.softmax(outputs, 1)
                _, preds = torch.max(probs, 1)
                f1_scores = calculate_metrics(preds, masks)
                val_f1_p += f1_scores[1][0]
                val_f1_qrs += f1_scores[2][0]
                val_f1_t += f1_scores[3][0]
                val_loss += criterion(outputs, masks).item()
        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1_p = val_f1_p / len(val_loader)
        avg_val_f1_qrs = val_f1_qrs / len(val_loader)
        avg_val_f1_t = val_f1_t / len(val_loader)
        channel_history['train'][1].append(avg_train_f1_p)
        channel_history['train'][2].append(avg_train_f1_qrs)
        channel_history['train'][3].append(avg_train_f1_t)
        channel_history['val'][1].append(avg_val_f1_p)
        channel_history['val'][2].append(avg_val_f1_qrs)
        channel_history['val'][3].append(avg_val_f1_t)
        print(f'Channel {channel.upper()} | Epoch {epoch+1}')
        print(f'Loss: {avg_train_loss:.4f} (train) | {avg_val_loss:.4f} (val)')
        print(f'F1 P: {avg_train_f1_p:.4f} (train) | {avg_val_f1_p:.4f} (val)')
        print(f'F1 QRS: {avg_train_f1_qrs:.4f} (train) | {avg_val_f1_qrs:.4f} (val)')
        print(f'F1 T: {avg_train_f1_t:.4f} (train) | {avg_val_f1_t:.4f} (val)')
        if avg_val_f1_p > best_val_f1_p:
            best_val_f1_p = avg_val_f1_p
            counter = 0
            torch.save(model.state_dict(), f'best_unet_{channel}.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Ранняя остановка!")
                break
    model.load_state_dict(torch.load(f'best_unet_{channel}.pth'))
    model.eval()
    test_f1_p = 0.0
    test_f1_qrs = 0.0
    test_f1_t = 0.0
    with torch.no_grad():
        for signals, masks in test_loader:
            signals, masks = signals.to(device), masks.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, 1)
            _, preds = torch.max(probs, 1)
            f1_scores = calculate_metrics(preds, masks)
            test_f1_p += f1_scores[1][0]
            test_f1_qrs += f1_scores[2][0]
            test_f1_t += f1_scores[3][0]
    print(f'\n*** Результаты на тестовой выборке для канала {channel.upper()} ***')
    print(f'Test F1 P: {test_f1_p / len(test_loader):.4f}')
    print(f'Test F1 QRS: {test_f1_qrs / len(test_loader):.4f}')
    print(f'Test F1 T: {test_f1_t / len(test_loader):.4f}')
    plot_metrics_history(channel_history, channel)

def visualize_results(model, dataset, channel, num_samples=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 5 * num_samples))
    for i in range(num_samples):
        signal, mask = dataset[i]
        signal, mask = signal.unsqueeze(0).to(device), mask.to(device)
        with torch.no_grad():
            output = model(signal)
            probs = torch.softmax(output, 1)
            pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        signal = signal.squeeze().cpu().numpy()
        mask = mask.cpu().numpy()
        time = np.linspace(0, record_duration, expected_samples)
        def plot_wave_regions(ax, indices, label, color, alpha=0.3):
            intervals = []
            start = None
            for j in range(len(indices)):
                if indices[j] and (start is None):
                    start = j
                elif not indices[j] and start is not None:
                    intervals.append((start, j - 1))
                    start = None
            if start is not None:
                intervals.append((start, len(indices) - 1))
            for start_idx, end_idx in intervals:
                ax.fill_between(
                    time[start_idx:end_idx + 1],
                    signal[start_idx:end_idx + 1].min(),
                    signal[start_idx:end_idx + 1].max(),
                    color=color,
                    alpha=alpha,
                    label=label
                )
                label = None
        ax_true = axes[0] if num_samples == 1 else axes[i][0]
        ax_true.plot(time, signal, label='Сигнал ЭКГ', color='black', linewidth=1)
        plot_wave_regions(ax_true, mask == 1, 'P (истинная)', 'blue')
        plot_wave_regions(ax_true, mask == 2, 'QRS (истинная)', 'red')
        plot_wave_regions(ax_true, mask == 3, 'T (истинная)', 'green')
        ax_true.set_title(f'Истинная разметка - Пример')
        ax_true.legend()
        ax_true.grid(True)
        ax_true.set_xlabel('Время (секунды)')
        ax_true.set_ylabel('Амплитуда')
        ax_pred = axes[1] if num_samples == 1 else axes[i][1]
        ax_pred.plot(time, signal, label='Сигнал ЭКГ', color='black', linewidth=1)
        plot_wave_regions(ax_pred, pred == 1, 'P (предсказанная)', 'cyan', alpha=0.2)
        plot_wave_regions(ax_pred, pred == 2, 'QRS (предсказанная)', 'orange', alpha=0.2)
        plot_wave_regions(ax_pred, pred == 3, 'T (предсказанная)', 'purple', alpha=0.2)
        ax_pred.set_title(f'Предсказанная разметка - Пример')
        ax_pred.legend()
        ax_pred.grid(True)
        ax_pred.set_xlabel('Время (секунды)')
        ax_pred.set_ylabel('Амплитуда')
    plt.tight_layout()
    plt.show()

print("\n*** Поиск оптимальных гиперпараметров с помощью Optuna ***")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)

print("\n*** Лучшие параметры ***")
trial = study.best_trial
print(f"Значение: {trial.value}")
for key, value in trial.params.items():
    print(f"{key}: {value}")

best_params = trial.params
all_channels = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
print("\n*** Обучение всех каналов с оптимальными параметрами ***")
for channel in all_channels:
    print(f"\n*** Обучение для канала {channel.upper()} ***")
    train(channel, use_best_params=True, best_params=best_params)

print("\n*** Визуализация результатов ***")
for channel in all_channels:
    print(f"\n*** Визуализация для канала {channel.upper()} ***")
    model = UNet1D(num_classes=4).to('cuda')
    model.load_state_dict(torch.load(f'best_unet_{channel}.pth'))
    model.eval()
    test_dataset = ECGDataset(
        records=test_records,
        channels=[channel],
        data_path=data_path,
        fs=fs,
        expected_samples=expected_samples,
        classes_config=classes_config
    )
    #Визуализация результатов
    visualize_results(model, test_dataset, channel, num_samples=5)
    ''''lr': 0.0008074737865207043, 'weight_decay': 0.00018498948148672935, 'dropout_rate': 0.10390800133578551, 'batch_size': 4}'''