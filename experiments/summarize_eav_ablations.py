#!/usr/bin/env python3
"""Evaluate saved ablation students and write paper-ready EAV tables."""
import importlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(ROOT / "archieve"))
import dataset.dataset as dataset_module
from KDbaseline.model.Student import ST_GCLSTM

DATA = {'eeg': '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'audio': '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace'}


def load_test_split(split):
    os.environ['EAV_SPLIT_IDX'] = str(split)
    importlib.reload(dataset_module)
    manager = dataset_module.CrossSubjectMultiModalDataset(DATA, audio_feature_type='opensmile', vision_feature_type='openface')
    _, _, (eeg, labels, pcc) = manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)
    return DataLoader(TensorDataset(torch.as_tensor(eeg).float(), torch.as_tensor(pcc).float(),
                                   torch.as_tensor(labels).long()), batch_size=128)


def evaluate(checkpoint, loader, device):
    model = ST_GCLSTM(num_nodes=30, in_features=5, gcn_hidden=64, gcn_out=64,
                      lstm_hidden=64, lstm_layers=1, fc_hidden=64,
                      num_classes=5, dropout=.5).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state'])
    model.eval(); predictions=[]; targets=[]
    with torch.no_grad():
        for x, graph, y in loader:
            predictions.extend(model(x.to(device), graph.to(device))['logits'].argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    return {'acc': accuracy_score(targets, predictions),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0)}


def main():
    device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
    result = {}
    names = {'kd':'best_student_kd.pth', 'fitnets':'best_student_fitnets.pth', 'nst':'best_student_nst.pth'}
    result = {method: {} for method in names}
    for split in (3,37,25):
        loader = load_test_split(split)
        for method, filename in names.items():
            path = REPO / 'ablations/eav_supervision_target/eeg_t' / method / f'split{split}' / filename
            result[method][str(split)] = evaluate(path, loader, device)
    for method in names:
        values=list(result[method].values())
        result[method]['mean']={'acc':float(np.mean([x['acc'] for x in values])),
                                'f1':float(np.mean([x['f1'] for x in values]))}
    out=REPO/'results/eav_requested_ablations.json'; out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
