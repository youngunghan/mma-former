# How-to — 교차검증 실행·채널 선택·이어받기·산출 해석

> **범위:** [MMA-Former.py](../../MMA-Former.py)로 6-fold 교차검증을 돌리고, 입력 채널을 고르고, 체크포인트를 이어받고, 산출 CSV/플롯을 읽는 실무 절차. 설치·첫 실행은 [tutorials/quickstart.md](../tutorials/quickstart.md).
> **대상:** 실험 실행자.
> **상태:** 구현 반영 — 기준일 2026-06-27.

## 1. 6-fold 교차검증 전체 실행

스크립트는 **한 번에 한 fold**만 학습한다. 전체 교차검증은 `val_fold`를 0–5로 바꿔 6회 실행한다:

```bash
for f in 0 1 2 3 4 5; do
  python MMA-Former.py \
    --val_fold $f \
    --preprocessed_dir /path/to/preprocessed \
    --folds_csv /path/to/folds.csv \
    --random_seed 42 \
    --postfix cv
done
```

- 각 실행은 `fold{f}_seed42_..._cv/` 별도 디렉터리에 산출물을 남긴다.
- fold별 best AUC/F1을 모아 평균±표준편차로 보고하는 것이 표준. 집계 스크립트는 별도(저장소에 없음).
- ⚠️ AUC가 `nan`으로 찍히는 fold가 있으면 그 fold 검증셋이 단일 클래스일 수 있다([known-issues.md](../explanation/known-issues.md) #6).

## 2. 입력 채널 선택

`.npy`가 여러 채널을 담고 일부만 쓰려면:

```bash
python MMA-Former.py --val_fold 1 --selected_channels "0,2" ...
```

- `image[selected_channels]`로 슬라이싱되며 선택 개수가 `in_channels`로 모델에 전달된다(patch_embed 입력 채널이 그만큼 바뀜).
- 미지정 시 3채널 전체.

## 3. 체크포인트 이어받기 (주의)

저장 디렉터리에 `best_neonet_model.pth`가 있으면 자동으로 로드한다. **단, 이는 진짜 resume이 아니다:**

- weight만 `strict=False`로 로드 — optimizer·epoch·best_val_loss·scaler는 복원되지 않는다.
- 따라서 학습은 epoch 0, 새 옵티마이저, `best=inf`로 다시 시작한다(warm-start). early-stopping 카운터도 리셋.
- 🔴 **이력 소실 주의**: 이렇게 "계속 학습"해도 `neonet_training_metrics.csv`·`neonet_val_predictions.csv`는 무조건 새로 덮어써져 **이전 run의 metric/prediction 이력이 전부 사라진다**([known-issues.md](../explanation/known-issues.md) #14). best epoch 주석도 재시작 이후 구간만 반영. 이력 보존이 필요하면 미리 CSV를 백업하거나 코드에서 append 모드로 바꿔야 한다.
- 상세·함정: [known-issues.md](../explanation/known-issues.md) #4. 진짜 resume이 필요하면 코드에 optimizer/epoch 저장·복원을 추가해야 한다.

## 4. 산출 CSV/플롯 읽기

### `neonet_training_metrics.csv`
epoch별 한 줄: `train_loss, val_loss, val_acc, val_f1, val_dice, val_auc, load_balance_loss`.
- **모델 선택 기준은 `val_loss` 최저**(= BCE + 0.005·LB, 순수 BCE 아님 — #8).
- `val_dice`는 `val_f1`과 항상 같다(#7) → 무시 가능.

### `neonet_val_predictions.csv`
epoch·pid·gt·pred_score(시그모이드 확률)·pred_binary(0.5 임계). 환자 단위 오분류 분석에 사용.

### `pictures/neonet_metrics_epoch_*.png`
매 10 epoch(및 epoch 1) 저장되는 6분할 플롯(loss·auc·f1·acc·dice·종합). val_loss 최저점이 별표로 표시된다.
- ⚠️ loss 패널의 load-balance 곡선 범례는 `"Load Balance Loss (×100)"`이지만 실제로는 **미스케일 plot**이라 범례가 크기를 100배 과대 표기한다([known-issues.md](../explanation/known-issues.md) #15). 실제 값은 CSV의 `load_balance_loss` 컬럼을 본다.

## 5. 재현성

- `--random_seed`가 `set_determinism`(MONAI)+`random`+`numpy`+`torch`+`torch.cuda`를 시드한다(단, `torch.cuda.manual_seed`만 — `manual_seed_all`은 미사용).
- ⚠️ DataLoader가 `worker_init_fn`·`generator`를 지정하지 않아 `num_workers>0`의 forked 워커에서 MONAI 증강 RNG가 시드에 고정된다고 보장할 수 없다(`set_determinism`만으로 fork 워커 결정성을 보장한다는 통념은 부정확). cuDNN 결정성 플래그도 설정하지 않는다.
- AMP는 항상 켜져 있어 끌 수 없다(#5) → 완전한 비트 재현이 필요하면 위 항목을 코드에서 보강해야 한다([known-issues.md](../explanation/known-issues.md) #16).

## 관련 문서

- [how-to/tune-moh.md](tune-moh.md) — MoH 하이퍼파라미터.
- [reference/configuration.md](../reference/configuration.md) — 전체 인자.
- [explanation/known-issues.md](../explanation/known-issues.md) — 실행 전 주의.
