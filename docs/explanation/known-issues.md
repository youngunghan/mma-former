# 알려진 이슈 — 코드 검증 결함·함정

> **범위:** [MMA-Former.py](../../MMA-Former.py)를 코드 대조·2-에이전트 적대 검증으로 확인한 결함·함정을 심각도별로 정리한다. **학습 결과 해석·재사용·논문화 전 필독.** 본 문서는 현행 코드의 *as-is* 기록이며, 수정 제안은 명시한다(자동 수정 아님).
> **대상:** 모델을 학습·평가·인용하는 모든 사람.
> **상태:** 코드 검증 기준일 2026-06-27. 마커: 🔴 치명 · 🟠 조건부/주의 · 🟢 의도된 제한/경미.

## 요약 표

| # | 심각도 | 위치(심볼) | 한 줄 |
|---|---|---|---|
| 1 | 🔴 | `MoHWindowAttention3D.forward` `dim_proj` | 6개 lazy `Linear`가 옵티마이저에 없어 **영구 미학습**(random init 고정), 전 어텐션 출력이 이를 통과 |
| 2 | 🟠 | `MoHLGTBlock.__init__` `drop_path` | stochastic depth가 항상 no-op(ternary 양 분기 `Identity`) |
| 3 | 🟠 | `MoHWindowAttention3D.forward` | MoH 경로(합산→`head_dim`)와 dense 경로(concat→`C`)의 차원 처리 불일치 |
| 4 | 🟠 | `__main__` 체크포인트 로딩 | "이어받기"가 weight만 복원 — optimizer·epoch·scaler·best_loss 미복원, epoch 0부터 재시작 |
| 5 | 🟠 | `--use_mixed_precision` | `store_true`+`default=True`라 CLI로 끌 수 없음 |
| 6 | 🟠 | `compute_metrics` AUC | bare `except`로 단일 클래스 fold 시 조용히 `nan` |
| 7 | 🟢 | `compute_metrics` `dice` | 이진 분류에서 `f1`과 수학적으로 동일(중복 메트릭) |
| 8 | 🟢 | best-model 기준 | val_loss = BCE + 0.005·load_balance라 순수 BCE가 아님(모델 선택이 보조손실에 영향) |
| 9 | 🟢 | MoH 라우팅 단위 | 토큰 단위가 아니라 **윈도우 단위**(표준 MoH와 상이) |
| 10 | 🟢 | `SAF.forward` | 고해상 `feat_b`를 다운샘플 후 융합(고해상 정보 손실 가능) + 양방향 `cross_attn` 공유 |
| 11 | 🟢 | 경로·전처리 가정 | `/home/rintern07/...` 하드코딩, 96×96×48·3채널 강결합, fold 범위 밖 행 무음 누락, 중복 ID 행 처리 |
| 12 | 🟢 | `torch._dynamo.suppress_errors=True` | dynamo 컴파일 실패를 조용히 무시 |
| 13 | 🟠 | `MoHLGTBlock.__init__` `shift_size` | 저장만 되고 `forward`에서 미사용 — Swin식 shifted-window 미구현(`drop_path` no-op와 동형) |
| 14 | 🟠 | `__main__` 재시작 시 CSV | 이어받기 시 metrics/predictions CSV를 무조건 `"w"`로 재오픈 → **이전 학습 이력 전부 truncate**(데이터 손실) |
| 15 | 🟢 | `plot_metrics` 범례 | load-balance 곡선 범례가 "(×100)"인데 실제로는 미스케일 plot → 범례가 100배 과대 표기 |
| 16 | 🟢 | 재현성 한계 | `worker_init_fn`/`generator` 없음, cuDNN 결정성 플래그 미설정, AMP 항상 on → `num_workers>0`에서 완전 비트재현 불가 |

---

## 1. 🔴 `dim_proj` 미학습 — 가장 심각

기본 설정(헤드 6/8/12, `num_shared_heads=2`)에서 모든 어텐션이 MoH 경로를 탄다. 이 경로는 shared·routed 헤드 출력을 **합산**해 마지막 차원을 `head_dim`(8/12/16)으로 붕괴시키므로, `combined_output.size(-1) != C`가 항상 참이 되어 다음이 실행된다:

```python
if self.dim_proj is None:
    self.dim_proj = nn.Linear(combined_output.size(-1), C).to(x.device)
combined_output = self.dim_proj(combined_output)
```

- `dim_proj`는 `__init__`이 아니라 **첫 forward에서 lazy 생성**된다.
- 그런데 옵티마이저는 학습 루프 진입 *전에* `AdamW(model.parameters())`로 만들어진다. 그 시점 `dim_proj`는 아직 `None` → 파라미터가 옵티마이저 param group에 **포함되지 않음**.
- 결과: `attn_local`·`attn_global` × 3블록 = **6개의 `dim_proj`가 random 초기값으로 고정**되어 학습 내내 갱신되지 않는다. 그런데 모든 어텐션 출력이 이 고정 random projection을 통과한다.

**영향:** 각 윈도우 어텐션이 학습 불가능한 무작위 선형사상으로 압축된다. 모델이 그 주변(qkv/proj)으로 보정 학습할 수는 있으나, 명백히 의도치 않은 동작이며 표현력 손실·재현성 저하의 큰 원인이다.

**부가 함정:** 저장된 체크포인트에는 (첫 forward 후 등록되므로) `dim_proj.*` 키가 들어가지만, 이어받기 시 `strict=False`로 *unexpected key*로 조용히 버려지고 첫 forward에서 다시 random 생성된다(#4와 결합).

**수정 방향(제안):**
- `dim_proj`를 `__init__`에서 미리 생성하거나, 더 근본적으로 헤드를 **합산하지 말고 concat**해 `C`를 유지하면 `dim_proj` 자체가 불필요(dense 경로와 일관). 어느 쪽이든 옵티마이저 생성 전에 모든 파라미터가 존재해야 한다.

## 2. 🟠 `drop_path` no-op

```python
self.drop_path = nn.Identity() if drop_path == 0. else nn.Identity()
```

양 분기가 모두 `Identity`라 `drop_path` 인자가 무엇이든 stochastic depth가 적용되지 않는다. NeoNet은 `drop_path`를 넘기지도 않아 어차피 0이지만, 값 변경으로 정규화를 켜려 해도 무효다. → `DropPath`(timm류) 구현으로 교체 필요.

## 3. 🟠 MoH/dense 경로 차원 불일치

- MoH 경로(`num_routed_heads>0`): 헤드 **합산** → `head_dim` → `dim_proj`로 복원.
- dense 경로(`num_routed_heads==0`): `transpose(1,2).view(B_,N,-1)`로 헤드 **concat** → `C` 유지.

두 경로가 출력 차원을 다르게 만든다. 기본 설정에선 dense 경로가 dead code라 충돌은 안 나지만, `num_shared_heads ≥ num_heads`로 바꾸면 동작이 질적으로 달라진다. 설계상 두 경로는 같은 의미를 가져야 하므로 합산↔concat 통일 권장.

## 4. 🟠 부분적 체크포인트 "이어받기"

`best_neonet_model.pth`가 있으면 로드하지만:

```python
checkpoint = torch.load(best_model_path, ...)
model.load_state_dict(checkpoint, strict=False)
```

- 복원 대상은 **모델 weight뿐**. optimizer 상태·`scaler`·`epoch`·`best_val_loss`·`epochs_since_improvement`는 복원하지 않는다.
- 학습 루프는 항상 `epoch 0`, `best_val_loss=inf`, 새 옵티마이저로 시작 → 진짜 resume이 아니라 **weight warm-start**다. early-stopping·LR 스케줄 관점에서 처음부터 다시 센다.
- `strict=False`라 형상/키 불일치가 조용히 무시된다(#1의 `dim_proj` 키 포함).
- ⚠️ 게다가 이어받기로 "계속 학습"해도 metrics/predictions CSV는 무조건 새로 덮어써진다 → **이전 이력이 전부 사라진다**(#14 참조).

## 5. 🟠 AMP를 끌 수 없음

`--use_mixed_precision`이 `action="store_true", default=True`라 항상 `True`다. mixed precision을 비활성화하려면 코드 수정이 필요(예: `--no_mixed_precision` 추가 또는 `BooleanOptionalAction`).

## 6. 🟠 AUC 무음 NaN

```python
try:
    auc = roc_auc_score(trues, preds)
except:
    auc = float('nan')
```

bare `except`라 검증 fold에 한 클래스만 있을 때(또는 다른 예외) AUC가 조용히 `nan`이 된다. 작은 fold(168/6 ≈ 28명)에서 클래스 불균형 시 발생 가능 — 로그에 경고 없이 묻힌다. 예외 종류를 좁히고 경고 출력 권장.

## 7. 🟢 `dice` == `f1` 중복

`dice = 2·Σ(pred_bin·true)/(Σpred_bin+Σtrue)`는 이진 라벨에서 F1과 동일한 값이다. 두 메트릭이 항상 같은 수를 보고하므로 `dice`는 정보가 없다(분할이 아닌 분류 태스크의 흔적으로 보임).

## 8. 🟢 best-model 기준이 순수 BCE가 아님

early-stopping과 best 저장은 `avg_val_loss` 최저 기준인데, 이 값은 `BCE + 0.005·load_balance_loss`다. 모델 선택이 보조손실(라우터 균형)에 미세하게 좌우된다. 분류 성능만으로 고르려면 AUC 또는 순수 BCE 기준이 더 적절할 수 있다.

## 9. 🟢 윈도우 단위 라우팅

라우터 입력 `window_repr = x.mean(dim=1)`은 윈도우당 1개라, 같은 윈도우의 모든 토큰이 동일 routed 헤드를 공유한다. 표준 MoH의 토큰 단위 라우팅과 다르다. 의도일 수 있으나(3D 윈도우 비용 절감) 명시적 설계 결정으로 기록할 가치가 있다.

## 10. 🟢 SAF 다운샘플·공유 cross_attn

- `SAF.forward`는 고해상 `feat_b`를 저해상 `feat_a` 크기로 줄인 뒤 융합 → 세밀 정보 손실 가능(U-Net skip의 통상 방향과 반대).
- 두 방향 cross-attention이 같은 `self.cross_attn` 인스턴스를 공유 → 비대칭 관계를 한 파라미터로 모델링.

둘 다 결함이라기보다 **설계 선택**이나, ablation 시 점검 포인트.

## 11. 🟢 환경 강결합·무음 누락

- 기본 경로가 `/home/rintern07/...`로 하드코딩 → 다른 환경에서 반드시 `--preprocessed_dir`·`--folds_csv` 지정.
- 입력 96×96×48·3채널에 강결합(patch_embed·`input_resolution`). 다른 크기는 형상 오류.
- fold 값이 `range(6)` 밖이거나 NaN, 또는 `.npy`가 없는 행은 train/val 어디에도 안 들어가고 **조용히 누락**(경고 없음) → 의도치 않은 표본 손실 위험.
- **중복 ID 행**: `load_preprocessed_data_with_folds`는 `folds_df` 행을 순회하고 `fold_dict`/`pni_dict`를 `dict(zip(ID, ...))`로 만든다. CSV에 같은 `ID`가 여러 번 있으면 (a) 같은 `.npy`가 train 리스트에 중복 추가되어 손실/메트릭이 편향되고 (b) `PNI`/`fold`는 **마지막 행 값으로 덮어써진다**. dedup·경고 없음 → manifest의 ID 유일성 가정. (`drop_duplicates(subset='ID')` 또는 `assert ID.is_unique` 권장.)

## 12. 🟢 dynamo 오류 억제

파일 상단 `torch._dynamo.config.suppress_errors = True`. load-balance loss의 Python for-loop·boolean 인덱싱이 `torch.compile` 그래프 캡처와 충돌하는데, 이 설정이 컴파일 실패를 eager로 조용히 폴백시킨다 — 성능 저하가 숨겨질 수 있다.

## 13. 🟠 `shift_size` 미사용 — shifted-window 미구현

`MoHLGTBlock.__init__`이 `shift_size` 인자를 받아 `self.shift_size`에 저장하지만 `forward` 어디에서도 쓰지 않는다(`torch.roll`·윈도우 시프트·attention mask 로직 자체가 없음). 즉 Swin류 **shifted-window attention이 구현되지 않았고**, `shift_size`에 어떤 값을 넣어도 무효다(#2 `drop_path` no-op와 정확히 동형). NeoNet은 이 인자를 넘기지도 않아 항상 0이지만, 시프트로 윈도우 간 정보 교환을 켜려 해도 동작하지 않는다 → `torch.roll` 기반 시프트+마스킹을 구현해야 한다.

## 14. 🟠 재시작 시 metrics/predictions CSV truncate (데이터 손실)

`best_neonet_model.pth`가 있어 weight warm-start(#4)로 "계속 학습"하는 경우에도, 학습 시작부에서 `neonet_training_metrics.csv`·`neonet_val_predictions.csv`를 **무조건 `open(..., "w")`로 다시 연다**. 이는 resume 블록 뒤에서 조건 없이 실행되므로 **이전 run의 전체 metric·prediction 이력이 truncate(소실)**된다. 결과적으로 `plot_metrics`의 `idxmin` best-epoch 주석·`best_scores` 행도 재시작 이후 구간만으로 계산되어 **실제 best epoch을 더 이상 반영하지 못한다**. → 이어받기 시 CSV를 append 모드로 열고(파일 없을 때만 헤더 기록) 처리해야 한다.

## 15. 🟢 plot 범례 "×100" 오표기

`plot_metrics`가 load-balance 곡선을 그릴 때 범례를 `"Load Balance Loss (×100)"`로 달지만, 실제 plot되는 `metrics_df['load_balance_loss']`에는 ×100 스케일이 적용되지 않는다. 저장된 PNG를 읽는 사람은 load-balance 크기를 100배 과대 해석하게 된다 → 범례 문구를 고치거나 실제로 ×100 스케일을 적용해야 한다.

## 16. 🟢 재현성 한계

- DataLoader가 `worker_init_fn`·`generator`를 지정하지 않는다 → `num_workers>0`의 forked 워커에서 MONAI 증강 RNG가 시드에 고정되지 않을 수 있다(`set_determinism`만으로 fork 워커 결정성을 보장한다는 통념은 부정확).
- cuDNN 결정성 플래그(`torch.backends.cudnn.deterministic`/`benchmark`, `torch.use_deterministic_algorithms`)를 설정하지 않는다.
- `torch.cuda.manual_seed`(단일 device)만 쓰고 `manual_seed_all`은 쓰지 않는다.
- AMP가 항상 켜져 있어(#5) 완전한 비트 단위 재현은 어렵다. 엄밀 재현이 필요하면 위 항목을 코드에서 보강해야 한다.

## 관련 문서

- [explanation/moh-lgt.md](moh-lgt.md) — #1·#3·#9의 메커니즘.
- [explanation/saf-fusion.md](saf-fusion.md) — #10.
- [reference/configuration.md](../reference/configuration.md) — #5·#11 설정.
