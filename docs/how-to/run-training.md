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
- ⚠️ AUC가 `nan`으로 찍히는 fold가 있으면 그 fold 검증셋이 단일 클래스일 수 있다 — 이제 `[WARN] AUC undefined ...`가 함께 출력된다(✅ #6).
- `--output_dir ./runs`로 산출 루트를 옮기거나, `--config configs/example_config.json`으로 경로·하이퍼파라미터를 파일에서 줄 수 있다(명시 CLI 플래그가 우선).
- 데이터 manifest 이상(중복 ID·fold 범위·누락 파일)은 `[WARN]`으로 표시되고, `--strict_data`를 주면 곧바로 `ValueError`로 중단한다(✅ #11).

## 2. 입력 채널 선택

`.npy`가 여러 채널을 담고 일부만 쓰려면:

```bash
python MMA-Former.py --val_fold 1 --selected_channels "0,2" ...
```

- `image[selected_channels]`로 슬라이싱되며 선택 개수가 `in_channels`로 모델에 전달된다(patch_embed 입력 채널이 그만큼 바뀜).
- 미지정 시 3채널 전체.

## 3. 체크포인트 이어받기 (✅ 진짜 resume)

저장 디렉터리에 `best_neonet_model.pth`가 있으면 자동으로 로드해 **이어서 학습**한다(✅ 2026-06-27 개선).

- 체크포인트는 `model`+`optimizer`+`scaler`+`epoch`+`best_val_loss`+`best_val_loss_auc`+`epochs_since_improvement`를 모두 담아 저장·복원한다. 학습은 `start_epoch`부터 이어지고 best·early-stopping 카운터도 보존된다.
- metrics/predictions CSV는 resume일 때 **append**(이력 보존), fresh run일 때만 새로 쓴다([known-issues.md](../explanation/known-issues.md) #14). 따라서 best-epoch 주석도 전체 이력에서 계산된다.
- 구버전 bare `state_dict` 체크포인트는 자동 감지되어 weight-only warm-start로 로드된다(하위 호환). 단 **신형 체크포인트는 구코드로 못 읽는다**(저장 형식 변경).
- **dim_proj 복원(✅ 2026-06-27)**: `dim_proj`는 lazy 생성이라 로드 시점엔 모듈이 없어 저장된 `dim_proj.*`가 버려지고 forward에서 새 random projection이 생기던 문제가 있었다. 이제 **resume일 때만** 더미 forward 1회로 `dim_proj`를 먼저 materialize한 뒤 로드해, 저장된 random projection을 그대로 복원한다 → resume 재현성 확보. fresh run은 더미 forward를 돌리지 않아 동작 불변. (단 `dim_proj`는 여전히 옵티마이저 밖이라 **미학습**으로 보존 — #1.)

## 4. 산출 CSV/플롯 읽기

### `neonet_training_metrics.csv`
epoch별 한 줄: `train_loss, val_loss, val_acc, val_f1, val_dice, val_auc, load_balance_loss`.
- **모델 선택 기준은 `val_loss` 최저**(= BCE + 0.005·LB, 순수 BCE 아님 — #8).
- `val_dice`는 `val_f1`과 항상 같다(#7) → 무시 가능.

### `neonet_val_predictions.csv`
epoch·pid·gt·pred_score(시그모이드 확률)·pred_binary(0.5 임계). 환자 단위 오분류 분석에 사용.

### `pictures/neonet_metrics_epoch_*.png`
매 10 epoch(및 epoch 1) 저장되는 6분할 플롯(loss·auc·f1·acc·dice·종합). val_loss 최저점이 별표로 표시된다. load-balance 곡선 범례는 정정되었다(✅ #15, 이전엔 "(×100)" 오표기).

## 5. 재현성

- `--random_seed`가 `set_determinism`(MONAI)+`random`+`numpy`+`torch`+`torch.cuda`를 시드한다(단, `torch.cuda.manual_seed`만 — `manual_seed_all`은 미사용).
- ⚠️ DataLoader가 `worker_init_fn`·`generator`를 지정하지 않아 `num_workers>0`의 forked 워커에서 MONAI 증강 RNG가 시드에 고정된다고 보장할 수 없다(`set_determinism`만으로 fork 워커 결정성을 보장한다는 통념은 부정확). cuDNN 결정성 플래그도 설정하지 않는다([known-issues.md](../explanation/known-issues.md) #16 — 결과 변동 우려로 미수정 보존).
- AMP는 `--no-use_mixed_precision`으로 끌 수 있다(✅ #5). 단 #16의 다른 요인이 남아 있어 `num_workers>0`에서 완전한 비트 재현은 여전히 어렵다 — 엄밀 재현이 필요하면 워커 시드·cuDNN 플래그를 추가로 보강해야 한다.

## 관련 문서

- [how-to/tune-moh.md](tune-moh.md) — MoH 하이퍼파라미터.
- [reference/configuration.md](../reference/configuration.md) — 전체 인자.
- [explanation/known-issues.md](../explanation/known-issues.md) — 실행 전 주의.
