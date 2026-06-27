# Quickstart — 단일 fold 학습 돌려보기

> **범위:** 의존성 설치부터 단일 검증 fold로 [MMA-Former.py](../../MMA-Former.py)를 한 번 학습시키고 산출물을 확인하기까지의 happy path. 전처리 데이터(`.npy`)·fold CSV는 이미 준비됐다고 가정한다(준비 절차는 이 저장소 밖).
> **대상:** 처음 실행하는 사람.
> **상태:** 구현 반영 — 기준일 2026-06-27.

## 1. 사전 준비

- Python 3.9+, CUDA GPU 권장(AMP 기본 on, CPU도 동작은 함).
- 의존성: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `monai`.

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm monai
```

- 데이터 두 가지:
  1. 전처리 볼륨 디렉터리 — 환자별 `patient_{ID}.npy`(또는 `patient{ID}.npy`), 형상 `(3, 96, 96, 48)`.
  2. fold CSV — 컬럼 `ID`, `fold`(0–5), `PNI`(0/1). 스키마는 [reference/data-model.md](../reference/data-model.md) §2.

## 2. 학습 실행

기본 경로는 `/home/rintern07/...`로 박혀 있으니 **자신의 경로로 덮어쓴다**(`--output_dir`로 산출 위치도 옮긴다):

```bash
python MMA-Former.py \
  --val_fold 1 \
  --preprocessed_dir /path/to/preprocessed_96x96x48 \
  --folds_csv /path/to/folds.csv \
  --output_dir ./runs \
  --batch_size 4 \
  --learning_rate 8e-5 \
  --random_seed 42
```

- `--val_fold 1` → fold 1이 검증, 나머지(0,2,3,4,5)가 학습.
- 입력 채널이 3이 아니면 `--selected_channels "0,1"`처럼 지정(개수가 `in_channels`가 됨).
- 경로·하이퍼파라미터를 파일로 관리하려면 [configs/example_config.json](../../configs/example_config.json)을 복사해 `--config my_config.json`으로 넘긴다(명시 CLI 플래그가 우선).
- GPU가 없으면 AMP는 자동으로 꺼진다(`--no-use_mixed_precision`으로 명시적 비활성도 가능).

## 3. 콘솔에서 확인할 것

시작 시 설정 배너와 파라미터 수가 찍히고, epoch마다 다음이 출력된다:

```text
Epoch   1 | Train: 0.7xxx (BCE: .., LB: ..) | Val: 0.6xxx (BCE: .., LB: ..) | AUC: 0.5xxx F1: .. Acc: ..
    ✅ New best model saved! Val Loss: 0.6xxx, AUC: 0.5xxx
```

- `LB`는 MoH load-balance 보조손실. `Val Loss`가 최저를 갱신할 때만 best 모델이 저장된다.
- early stop은 val_loss가 30 epoch 연속 개선 없을 때 발동(최대 200 epoch).

## 4. 산출물

`/home/rintern07/final/training/neonet/<run_name>/`(또는 해당 루트) 아래:

| 파일 | 내용 |
|---|---|
| `best_neonet_model.pth` | val_loss 최저 시점 체크포인트 dict(`model`·`optimizer`·`scaler`·`epoch`·best 상태) — 이어받기 입력 |
| `neonet_training_metrics.csv` | epoch별 loss·acc·f1·dice·auc·lb |
| `neonet_val_predictions.csv` | epoch·pid·gt·pred_score·pred_binary |
| `pictures/*.png` | 매 10 epoch 6분할 메트릭 플롯 |

## 5. 다음 단계

- 6-fold 전체 교차검증·채널 선택·이어받기 → [how-to/run-training.md](../how-to/run-training.md).
- MoH 하이퍼파라미터 튜닝 → [how-to/tune-moh.md](../how-to/tune-moh.md).
- ⚠️ 결과를 해석하기 전 [explanation/known-issues.md](../explanation/known-issues.md)를 먼저 읽어라(특히 🔴 `dim_proj` 미학습).
