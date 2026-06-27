# 설정 — CLI 인자·하드코딩 하이퍼파라미터·경로

> **범위:** [MMA-Former.py](../../MMA-Former.py) `__main__`의 `argparse` 인자와 코드에 하드코딩된 학습 상수·경로를 표로 정리한다. 설정은 CLI 인자 + 선택적 JSON 파일(`--config`, ✅ 2026-06-27)로 주거나 소스 상수로 박혀 있다.
> **대상:** 실험 실행자.
> **상태:** 구현 반영 — 기준일 2026-06-27.

## 1. CLI 인자

`python MMA-Former.py [옵션]`. 출처: `argparse.ArgumentParser`.

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--val_fold` | int | `1` | 검증 fold 인덱스. 나머지 fold(0–5 중)가 학습 |
| `--preprocessed_dir` | str | `/home/rintern07/neonet/training/preprocessed_all_96x96x48_FIXED` | `.npy` 볼륨 디렉터리 |
| `--folds_csv` | str | `/home/rintern07/final/data/new_168_fold.csv` | fold 정보 CSV(컬럼 ID·fold·PNI) |
| `--learning_rate` | float | `8e-5` | AdamW 학습률 |
| `--random_seed` | int | `42` | `set_determinism`+`random`+`numpy`+`torch`+`cuda` 시드 |
| `--batch_size` | int | `4` | train/val 공통 |
| `--num_workers` | int | `2` | DataLoader 워커 수 |
| `--use_mixed_precision` | bool flag | `True` | `BooleanOptionalAction` — `--no-use_mixed_precision`로 끌 수 있음. CPU에서는 자동 비활성(✅ 2026-06-27) |
| `--moh_efficiency` | float | `0.75` | MoH top-k 비율(아래 §2 참조) |
| `--load_balance_weight` | float | `0.005` | 총손실에서 load-balance loss 가중치 |
| `--num_shared_heads` | int | `2` | MoH always-on 공유 헤드 수 |
| `--postfix` | str | `fixed_saf` | 저장 디렉터리명 접미사 |
| `--selected_channels` | str | `None` | `"0,1,2"` 형태. 지정 시 `in_channels`가 선택 수로 결정, 미지정 시 3 |
| `--output_dir` | str | `/home/rintern07/final/training` | 산출 루트(체크포인트·metrics·plot). 기본은 기존값 유지(✅ 2026-06-27) |
| `--config` | str | `None` | JSON 설정 파일. 값이 인자 기본값이 되고 명시 CLI 플래그가 우선([configs/example_config.json](../../configs/example_config.json))(✅ 2026-06-27) |
| `--strict_data` | flag | `False` | 중복 ID/`range(6)` 밖 fold/누락 `.npy`를 경고 대신 `ValueError`로 중단(✅ 2026-06-27) |

## 2. MoH 파생값

`MoHWindowAttention3D.__init__`에서 계산된다. 튜닝 영향은 [how-to/tune-moh.md](../how-to/tune-moh.md) 참조.

| 파생값 | 계산식 | 기본 설정 결과 |
|---|---|---|
| `num_shared_heads` | `min(num_shared_heads, num_heads)` | 2 |
| `num_routed_heads` | `num_heads - num_shared_heads` | block1=4, block2=6, block3=10 |
| `top_k` | `max(1, int(num_routed_heads * moh_efficiency))` | block1=3, block2=4, block3=7 |

- `num_shared_heads ≥ num_heads`이면 `num_routed_heads=0`, `top_k=0` → MoH 라우팅이 꺼지고 dense 어텐션 경로로 빠진다. 기본 설정(헤드 6/8/12, 공유 2)에서는 **항상 MoH 경로**다.

## 3. 하드코딩 학습 상수

CLI로 노출되지 않고 소스에 박혀 있다. 변경하려면 코드 수정 필요.

| 상수 | 값 | 위치(심볼) |
|---|---|---|
| `max_epochs` | `200` | `__main__` |
| `early_stop_patience` | `30` | `__main__`(val_loss 기준) |
| optimizer | `AdamW(weight_decay=0.01)` | `__main__` |
| BCE `pos_weight` | `1.5` | `combined_loss_fn`(하드코딩) |
| classifier dropout | `0.15` | `NeoNet.__init__` |
| MLP `mlp_ratio` | `2.0` | 모든 `MoHLGTBlock` |
| classifier hidden dim | `64` | `NeoNet.__init__` |
| patch embed | `Conv3d(in, 48, k=4, s=4)` | `NeoNet.__init__` |
| 블록 차원 | 48 → 96 → 192 | `NeoNet.__init__` |
| 블록 헤드 수 | 6 → 8 → 12 | `NeoNet.__init__` |
| local 윈도우 | `(3,3,3)` 전 블록 | `NeoNet.__init__` |
| global 윈도우 | block1·2 `(6,6,6)`, block3 `(6,6,3)` | `NeoNet.__init__` |
| AMP `GradScaler` | `torch.amp` 우선, 실패 시 `torch.cuda.amp` | `__main__` |
| 플롯/best-score 저장 주기 | 매 10 epoch(또는 epoch 1) | `plot_metrics`·`save_best_scores_to_csv` |

## 4. 데이터 증강 (학습 전용)

MONAI dict transform. validation에는 적용 안 함(`transform=None`). 키는 `"input"`.

| Transform | 확률 | 파라미터 |
|---|---|---|
| `RandFlipd` | 0.5 | `spatial_axis=0` |
| `RandRotate90d` | 0.5 | `max_k=3` |
| `RandGaussianNoised` | 0.2 | `std=0.1` |
| `RandAffined` | 0.2 | rotate `0.1`, translate `[5,5,2]`, scale `0.05`, `mode="nearest"` |
| `RandShiftIntensityd` | 0.2 | `offsets=0.1` |

## 5. 산출 경로·파일

기준 루트는 `--output_dir`(기본 `/home/rintern07/final/training`) 아래 `neonet/`. `--output_dir`로 옮길 수 있다(✅ 2026-06-27, 이전엔 하드코딩).

- 저장 디렉터리명: `fold{V}_seed{S}_lr{LR}_moh{MOH%}_lb{LB}_sh{SH}_{postfix}` 형태.
- `best_neonet_model.pth` — val_loss 최저 시 **체크포인트 dict**(`epoch`·`model`·`optimizer`·`scaler`·`best_val_loss`·`best_val_loss_auc`·`epochs_since_improvement`) 저장. 이어받기 입력으로 사용(✅ 2026-06-27, 이전엔 bare `state_dict`; 구버전 체크포인트는 weight-only로 자동 호환).
- `neonet_training_metrics.csv` — epoch별 train/val loss·acc·f1·dice·auc·load_balance_loss.
- `neonet_val_predictions.csv` — epoch·pid·gt·pred_score·pred_binary.
- `pictures/neonet_metrics_epoch_*.png` — 6분할 메트릭 플롯(매 10 epoch).
- `best_scores/neonet_best_scores_*.csv` — best epoch 스냅샷 누적.

## 관련 문서

- [how-to/run-training.md](../how-to/run-training.md) — 인자 조합 실행 예.
- [reference/data-model.md](data-model.md) — 입력 스키마.
