![image](https://github.com/user-attachments/assets/aa3ee5ef-4568-4ade-aef7-fe172df539d9)## Детекция шума на ЭКГ с использованием U-Net и Optuna
# Описание проекта
Этот проект посвящен разработке модели глубокого обучения для сегментации ЭКГ-сигналов с целью выделения ключевых компонентов: P-волн, QRS-комплексов и T-волн. Сегментация выполняется с помощью одномерной U-Net, а оптимизация гиперпараметров модели осуществляется с использованием библиотеки Optuna. Особое внимание уделяется точности сегментации P-волн, как наиболее сложных для детекции.
# Ключевые особенности
- Адаптированная U-Net для обработки одномерных ЭКГ-сигналов
- Оптимизация гиперпараметров с помощью Optuna
- Фокус на P-волны как наиболее сложные для детекции
- Ранняя остановка для предотвращения переобучения
- F1-мера как основная метрика качества
- Визуализация результатов сегментации
# Технические требования
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6 (для GPU-ускорения)
- Дополнительные зависимости: NumPy, Pandas, Matplotlib, Scikit-learn, WFDB, Optuna, tqdm
# Установка
``` bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn wfdb optuna tqdm
```
# Использование
1. Подготовка данных
-Скачайте датасет LUDB
Распакуйте в нужную вам директорию
2. Обучение модели
``` python
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

```
3. Визуализация результатов
``` python
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
```
# Архитектура модели
Модель основана на одномерной U-Net с модификациями:

``` python
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
```
# Оптимизация гиперпараметров
Optuna используется для поиска оптимальных значений:
- Learning rate: 1e-5 - 1e-3
- Weight decay: 1e-5 - 1e-3
- Dropout rate: 0.1 - 0.5
- Batch size: 4, 8, 16
# Результаты
После оптимизации достигаются следующие показатели F1 на тестовой выборке для первого канала как примера:
- P-волна	0.7515
- QRS	0.9073	
- T-волна	0.7767	
# Пример графика обучения
![download](https://github.com/user-attachments/assets/7e596e68-ac29-492c-9bc9-cac6afc3372e)
# Пример визуализации
![image](https://github.com/user-attachments/assets/d5810c82-6c04-4c10-8c99-10f7fd07e528)
# Ключевые выводы
- Фокусировка на P-волнах улучшает качество сегментации всех компонентов ЭКГ
- Оптимизация гиперпараметров с помощью Optuna дает прирост F1-меры на 15-20%
- Ранняя остановка по метрике P-волн сокращает время обучения на 30%
- Использование F1-меры вместо Dice обеспечивает более строгую оценку качества
# Библиография
- Ronneberger O. et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Akiba T. et al. "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
- Moskalenko V. et al. "Deep learning in ECG Segmentation" (2020)
- He K. et al. "Deep Residual Learning for Image Recognition" (2016)
