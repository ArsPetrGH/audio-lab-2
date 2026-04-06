import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from itertools import permutations
from huggingface_hub import login
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.audio.pit import PermutationInvariantTraining as DiarizationErrorRate

torch.set_float32_matmul_precision('high')

# 1. Авторизація
HF_TOKEN = "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"
login(token=HF_TOKEN)


class AMIDataset(Dataset):
    def __init__(self, hf_dataset, epoch_samples=5000):
        self.samples = hf_dataset
        self.epoch_samples = epoch_samples
        self.unique_speakers = sorted(list(set(self.samples['main_speaker'])))
        self.spk_to_indices = {spk: [] for spk in self.unique_speakers}
        for idx, spk in enumerate(self.samples['main_speaker']):
            self.spk_to_indices[spk].append(idx)

    def __len__(self):
        return min(self.epoch_samples, len(self.samples))

    def make_vad_mask(self, vad_labels, n_frames=150, segment_len=3.0):
        mask = torch.zeros((4, n_frames))
        sorted_labels = sorted(vad_labels, key=lambda x: x['start'])
        for i, label in enumerate(sorted_labels[:4]):
            s_f = int(label['start'] / segment_len * n_frames)
            e_f = int(label['end'] / segment_len * n_frames)
            mask[i, max(0, s_f):min(n_frames, e_f)] = 1
        return mask

    def __getitem__(self, idx):
        anchor_data = self.samples[idx]

        def augment(audio):
            audio = audio * (random.random() * 0.7 + 0.5)
            noise = torch.randn_like(audio) * 0.001
            return audio + noise

        chunk = torch.tensor(anchor_data['audio'], dtype=torch.float32)
        chunk = augment(chunk)
        anchor_spk = anchor_data['main_speaker']
        target_mask = self.make_vad_mask(anchor_data['vad_labels'])

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.spk_to_indices[anchor_spk])
        pos_chunk = augment(torch.tensor(self.samples[pos_idx]['audio'], dtype=torch.float32))

        neg_spk = anchor_spk
        while neg_spk == anchor_spk:
            neg_spk = random.choice(self.unique_speakers)
        neg_idx = random.choice(self.spk_to_indices[neg_spk])
        neg_chunk = augment(torch.tensor(self.samples[neg_idx]['audio'], dtype=torch.float32))

        return chunk, target_mask.T, pos_chunk, neg_chunk


# ==========================================
# 3. MODEL
# ==========================================
class FullAudioModel(pl.LightningModule):
    def __init__(self, n_speakers=4, n_total_speakers=50, lr=5e-4, sample_rate=16000, n_mels=64):
        super().__init__()
        self.save_hyperparameters()

        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(1024, 256, batch_first=True, bidirectional=True)
        self.segmentator = nn.Linear(512, n_speakers)
        self.embedder = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 64)
        )

        # Основні метрики
        self.vad_acc = BinaryAccuracy()
        self.vad_f1 = BinaryF1Score()

        # Справжня метрика діаризації (замість MultilabelAccuracy)
        self.der_metric = DiarizationErrorRate(
            metric_func=F.binary_cross_entropy_with_logits,
            eval_func="min"
        )

        self.mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db_transform = T.AmplitudeToDB()

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        x = self.mel_transform(x)
        x = self.db_transform(x)
        x = x.unsqueeze(1)
        b, c, f, t = x.shape
        x = self.enc(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        logits = self.segmentator(lstm_out)
        avg_pool = torch.mean(lstm_out, dim=1)
        embedding = self.embedder(avg_pool)
        return logits, embedding

    def pit_loss(self, preds, targets):
        perms = list(permutations(range(self.hparams.n_speakers)))
        losses = torch.stack([
            F.binary_cross_entropy_with_logits(preds, targets[:, :, list(p)], reduction='none').mean(dim=(1, 2))
            for p in perms
        ])
        return losses.min(dim=0).values.mean()

    def triplet_loss(self, anchor, positive, negative, margin=0.3):
        a, p, n = F.normalize(anchor, dim=1), F.normalize(positive, dim=1), F.normalize(negative, dim=1)
        pos_dist = F.pairwise_distance(a, p)
        neg_dist = F.pairwise_distance(a, n)
        return F.relu(pos_dist - neg_dist + margin).mean()

    def training_step(self, batch, batch_idx):
        specs, y_mask, pos_specs, neg_specs = batch
        batch_size = specs.shape[0]
        all_logits, all_embs = self(torch.cat([specs, pos_specs, neg_specs], dim=0))

        y_hat_mask = F.interpolate(all_logits[:batch_size].transpose(1, 2), size=y_mask.shape[1],
                                   mode='linear').transpose(1, 2)

        pit = self.pit_loss(y_hat_mask, y_mask)
        triplet = self.triplet_loss(all_embs[:batch_size], all_embs[batch_size:2 * batch_size],
                                    all_embs[2 * batch_size:])
        total_loss = pit + 0.5 * triplet

        self.log_dict({"train_loss": total_loss, "train_pit": pit, "train_triplet": triplet}, prog_bar=True)
        return total_loss

    def smooth_predictions(self, logits, window_size=5):
        x = logits.transpose(1, 2)
        x_smoothed = F.avg_pool1d(x, kernel_size=window_size, stride=1, padding=window_size // 2)
        return x_smoothed.transpose(1, 2)

    def validation_step(self, batch, batch_idx):
        specs, y_mask, pos_specs, neg_specs = batch
        batch_size = specs.shape[0]
        all_logits, all_embs = self(torch.cat([specs, pos_specs, neg_specs], dim=0))

        y_hat_mask = F.interpolate(all_logits[:batch_size].transpose(1, 2), size=y_mask.shape[1],
                                   mode='linear').transpose(1, 2)
        y_hat_mask = self.smooth_predictions(y_hat_mask)

        # Лоси
        pit = self.pit_loss(y_hat_mask, y_mask)
        triplet = self.triplet_loss(all_embs[:batch_size], all_embs[batch_size:2 * batch_size],
                                    all_embs[2 * batch_size:])

        # DER (через PIT)
        val_der = self.der_metric(y_hat_mask.transpose(1, 2), y_mask.transpose(1, 2))

        # VAD
        preds = torch.sigmoid(y_hat_mask)
        target_vad = (y_mask.sum(dim=-1) > 0).float()
        preds_vad = preds.max(dim=-1)[0]
        self.vad_acc(preds_vad, target_vad)
        self.vad_f1(preds_vad, target_vad)

        self.log_dict({
            "val_loss": pit + 0.5 * triplet,
            "val_der": val_der,
            "val_vad_acc": self.vad_acc,
            "val_vad_f1": self.vad_f1,
            "val_triplet": triplet
        }, prog_bar=True, on_epoch=True)
        return pit + 0.5 * triplet

    def on_validation_epoch_end(self):
        self.der_metric.reset()

    def test_step(self, batch, batch_idx):
        specs, y_mask, pos_specs, neg_specs = batch
        batch_size = specs.shape[0]
        all_logits, all_embs = self(torch.cat([specs, pos_specs, neg_specs], dim=0))
        y_hat_mask = F.interpolate(all_logits[:batch_size].transpose(1, 2), size=y_mask.shape[1],
                                   mode='linear').transpose(1, 2)

        preds = torch.sigmoid(y_hat_mask)
        self.log_dict({
            "test_vad_acc": self.vad_acc(preds.max(dim=-1)[0], (y_mask.sum(dim=-1) > 0).float()),
            "test_der": self.der_metric(y_hat_mask.transpose(1, 2), y_mask.transpose(1, 2))
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ==========================================
# 5. VERIFICATION
# ==========================================
@torch.no_grad()
def evaluate_speaker_system(model, dataset, device="cuda", num_enrollment=5):
    model.eval()
    model.to(device)
    reference_db = {}
    print(f"Building reference database (Multi-shot: {num_enrollment} samples per spk)...")
    for spk in dataset.unique_speakers:
        indices = dataset.spk_to_indices[spk]
        enroll_indices = indices[:min(num_enrollment, len(indices))]
        embs = []
        for idx in enroll_indices:
            audio = torch.tensor(dataset.samples[idx]['audio']).unsqueeze(0).to(device)
            _, emb = model(audio)
            embs.append(F.normalize(emb, dim=1))
        reference_db[spk] = F.normalize(torch.stack(embs).mean(dim=0), dim=1)

    correct_id, total_test = 0, 1000
    pos_sims, neg_sims = [], []
    for _ in range(total_test):
        test_idx = torch.randint(0, len(dataset.samples), (1,)).item()
        test_data = dataset.samples[test_idx]
        audio = torch.tensor(test_data['audio']).unsqueeze(0).to(device)
        _, test_emb = model(audio)
        test_emb = F.normalize(test_emb, dim=1)
        max_sim, best_spk = -1.0, None
        for spk, ref_emb in reference_db.items():
            sim = (test_emb @ ref_emb.T).item()
            if spk == test_data['main_speaker']:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)
            if sim > max_sim: max_sim, best_spk = sim, spk
        if best_spk == test_data['main_speaker']: correct_id += 1

    print(
        f"\n📊 РЕЗУЛЬТАТИ ТЕСТУ:\n✅ Ідентифікація: {correct_id / total_test:.2%}\n🤝 Positive: {sum(pos_sims) / len(pos_sims):.4f}\n👤 Negative: {sum(neg_sims) / len(neg_sims):.4f}\n")

def detect_device():
    if torch.cuda.is_available():
        return "cuda", "gpu", 1, "bf16-mixed"
    if torch.backends.mps.is_available():
        return "mps", "mps", 1, "32"
    return "cpu", "cpu", "auto", "32"

if __name__ == "__main__":
    device, accelerator, devices, precision = detect_device()
    raw_ds = load_from_disk("D:/Oleksii_Chyzhov/AudioML/ami_segmented")
    train_ds = AMIDataset(raw_ds['train'], epoch_samples=6400)
    val_ds = AMIDataset(raw_ds['validation'], epoch_samples=640)
    test_ds = AMIDataset(raw_ds['test'], epoch_samples=640)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=6, persistent_workers=True,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=6, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=6, persistent_workers=True, pin_memory=True)

    model = FullAudioModel(n_total_speakers=len(train_ds.unique_speakers))
    trainer = pl.Trainer(max_epochs=10, accelerator=accelerator, devices=devices, precision=precision,
                         logger=CSVLogger("logs", name="ami"), enable_checkpointing=True)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    evaluate_speaker_system(model, test_ds, device=device)