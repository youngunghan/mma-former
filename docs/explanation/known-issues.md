# 알려진 이슈 — 코드 검증 결함·함정

> **범위:** [MMA-Former.py](../../MMA-Former.py)를 코드 대조·다중 에이전트 적대 검증으로 확인한 결함·함정을 심각도별로 정리한다. **학습 결과 해석·재사용·논문화 전 필독.**
> **대상:** 모델을 학습·평가·인용하는 모든 사람.
> **상태:** 코드 검증 기준일 2026-06-27. 마커: 🔴 치명 · 🟠 조건부/주의 · 🟢 의도된 제한/경미 · ✅ 해결.
> **2026-06-27 안전 수정 반영:** 학습 *결과/아키텍처를 바꾸지 않는* 견고성·재현성 항목을 코드에서 수정했다(✅ 표기: #4·#5·#6·#11·#14·#15). 모델 동작을 바꾸는 항목(#1 dim_proj·#2 drop_path·#3·#13 shift_size 등)은 **의도적으로 보존**(기존 체크포인트/결과 호환). fresh run의 학습 결과는 수정 전과 비트 동일하다.

## 요약 표

| # | 심각도 | 위치(심볼) | 한 줄 |
|---|---|---|---|
| 1 | 🔴 | `MoHWindowAttention3D.forward` `dim_proj` | 6개 lazy `Linear`가 옵티마이저에 없어 **영구 미학습**(random init 고정), 전 어텐션 출력이 이를 통과 |
| 2 | 🟠 | `MoHLGTBlock.__init__` `drop_path` | stochastic depth가 항상 no-op(ternary 양 분기 `Identity`) |
| 3 | 🟠 | `MoHWindowAttention3D.forward` | MoH 경로(합산→`head_dim`)와 dense 경로(concat→`C`)의 차원 처리 불일치 |
| 4 | ✅ | `__main__` 체크포인트 로딩 | (구) weight만 복원 → **해결**: optimizer·scaler·epoch·best·early-stop 상태까지 저장·복원(legacy 체크포인트 호환) |
| 5 | ✅ | `--use_mixed_precision` | (구) CLI로 못 끔 → **해결**: `BooleanOptionalAction`(`--no-use_mixed_precision`) + CPU에서 자동 비활성 gate |
| 6 | ✅ | `compute_metrics` AUC | (구) bare `except` → **해결**: `except ValueError`로 좁히고 경고 출력(원인 노출) |
| 7 | 🟢 | `compute_metrics` `dice` | 이진 분류에서 `f1`과 수학적으로 동일(중복 메트릭) |
| 8 | 🟢 | best-model 기준 | val_loss = BCE + 0.005·load_balance라 순수 BCE가 아님(모델 선택이 보조손실에 영향) |
| 9 | 🟢 | MoH 라우팅 단위 | 토큰 단위가 아니라 **윈도우 단위**(표준 MoH와 상이) |
| 10 | 🟢 | `SAF.forward` | 고해상 `feat_b`를 다운샘플 후 융합(고해상 정보 손실 가능) + 양방향 `cross_attn` 공유 |
| 11 | ✅/🟢 | 경로·전처리 가정 | (구) `/home/rintern07/...` 하드코딩·무음 누락 → **부분 해결**: `--output_dir`·`--config`·데이터 검증(중복ID/fold범위/누락) 경고+`--strict_data`. 96×96×48·3채널 강결합은 잔존(🟢) |
| 12 | 🟢 | `torch._dynamo.suppress_errors=True` | dynamo 컴파일 실패를 조용히 무시 |
| 13 | 🟠 | `MoHLGTBlock.__init__` `shift_size` | 저장만 되고 `forward`에서 미사용 — Swin식 shifted-window 미구현(`drop_path` no-op와 동형) |
| 14 | ✅ | `__main__` 재시작 시 CSV | (구) 재시작 시 CSV truncate → **해결**: resume 시 append(헤더는 파일 없을 때만), fresh run은 기존대로 truncate |
| 15 | ✅ | `plot_metrics` 범례 | (구) "(×100)" 오표기 → **해결**: 범례를 "Load Balance Loss"로 정정(실제 미스케일) |
| 16 | 🟢 | 재현성 한계 | `worker_init_fn`/`generator` 없음, cuDNN 결정성 플래그 미설정, `manual_seed_all` 미사용 → `num_workers>0`에서 완전 비트재현 불가(수정 시 결과 변동 우려로 보존; AMP는 ✅ #5로 끌 수 있음) |

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

**부가 함정(resume):** 저장된 체크포인트에는 `dim_proj.*` 키가 들어가지만, lazy 생성이라 로드 시점엔 모듈이 없어 *unexpected key*로 버려지고 첫 forward에서 새 random projection이 생기던 문제가 있었다 → **resume 재현성 훼손**. 이는 #4의 안전 수정에서 **resume 시 더미 forward로 dim_proj를 먼저 materialize한 뒤 로드**하도록 보완해 해소했다(저장된 random projection 복원). 단 이는 *resume 재현성*만 고친 것이고, dim_proj가 **여전히 옵티마이저 밖이라 학습되지 않는다는 본질 문제(아래)는 그대로**다.

**여전히 미해결(모델 동작 보존 — 사용자 선택):** 6개 `dim_proj`는 학습되지 않는다. 학습 *결과*를 바꾸는 변경이라 의도적으로 보존했다.

**수정 방향(제안 — 채택 시 모델 동작 변경·재학습 필요):**
- `dim_proj`를 `__init__`에서 미리 생성(옵티마이저에 포함)하거나, 더 근본적으로 헤드를 **합산하지 말고 concat**해 `C`를 유지하면 `dim_proj` 자체가 불필요(dense 경로와 일관). 어느 쪽이든 옵티마이저 생성 전에 모든 파라미터가 존재해야 한다. 이 변경은 기존 체크포인트/결과와 호환되지 않는다.

## 2. 🟠 `drop_path` no-op

```python
self.drop_path = nn.Identity() if drop_path == 0. else nn.Identity()
```

양 분기가 모두 `Identity`라 `drop_path` 인자가 무엇이든 stochastic depth가 적용되지 않는다. NeoNet은 `drop_path`를 넘기지도 않아 어차피 0이지만, 값 변경으로 정규화를 켜려 해도 무효다. → `DropPath`(timm류) 구현으로 교체 필요.

## 3. 🟠 MoH/dense 경로 차원 불일치

- MoH 경로(`num_routed_heads>0`): 헤드 **합산** → `head_dim` → `dim_proj`로 복원.
- dense 경로(`num_routed_heads==0`): `transpose(1,2).view(B_,N,-1)`로 헤드 **concat** → `C` 유지.

두 경로가 출력 차원을 다르게 만든다. 기본 설정에선 dense 경로가 dead code라 충돌은 안 나지만, `num_shared_heads ≥ num_heads`로 바꾸면 동작이 질적으로 달라진다. 설계상 두 경로는 같은 의미를 가져야 하므로 합산↔concat 통일 권장.

## 4. ✅ 체크포인트 "이어받기" (해결 2026-06-27)

(구) `best_neonet_model.pth`를 로드하되 **모델 weight만** 복원하고 optimizer·`scaler`·`epoch`·`best_val_loss`·`epochs_since_improvement`는 복원하지 않아, 항상 `epoch 0`·새 옵티마이저로 다시 시작하는 weight warm-start였다.

- **✅ 해결**: 체크포인트를 `{epoch, model, optimizer, scaler, best_val_loss, best_val_loss_auc, epochs_since_improvement}` dict로 저장하고, 로드 시 전부 복원해 `range(start_epoch, max_epochs)`로 이어간다. 구버전 bare `state_dict` 체크포인트는 자동 감지해 weight-only warm-start로 처리(하위 호환). 저장 형식이 바뀌므로 **구코드로는 신형 체크포인트를 못 읽는다**(신코드는 양쪽 모두 읽음).
- **✅ dim_proj 복원**: lazy `dim_proj` 때문에 저장된 random projection이 reload 시 버려지던 문제(#1 부가 함정)를 보완 — **resume일 때만** 더미 forward 1회로 dim_proj를 materialize한 뒤 `load_state_dict`해 저장값을 복원한다. fresh run은 더미 forward를 돌리지 않아 동작 불변. dim_proj는 여전히 옵티마이저 밖(미학습 보존).
- 🟢 잔존: `load_state_dict(strict=False)` 자체는 유지(다른 키 불일치 무음 허용). dim_proj 본질(미학습)은 #1로 보존.

## 5. ✅ AMP 토글 (해결 2026-06-27)

(구) `--use_mixed_precision`이 `store_true`+`default=True`라 항상 `True`였다.

- **✅ 해결**: `argparse.BooleanOptionalAction`으로 바꿔 `--no-use_mixed_precision`으로 끌 수 있다. 또한 `device.type != "cuda"`이면 AMP를 자동 비활성화해(경고 출력) CPU에서 `GradScaler('cuda')` 오용을 막는다. CUDA 기본 동작은 불변.

## 6. ✅ AUC 예외 처리 (해결 2026-06-27)

(구) bare `except:`라 단일 클래스 fold 등에서 AUC가 **조용히** `nan`이 되고 원인이 묻혔다.

- **✅ 해결**: `except ValueError as e:`로 좁히고 `[WARN] AUC undefined (...)`를 출력해 원인을 노출한다. 정상 케이스의 수치는 불변(여전히 단일 클래스면 `nan`이지만 경고가 찍힌다). 예기치 못한 다른 예외는 이제 숨기지 않고 전파된다.

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

## 11. ✅/🟢 환경 강결합·무음 누락 (부분 해결 2026-06-27)

- **✅ 출력 경로**: `--output_dir`(기본은 기존 하드코딩값 유지)로 산출 루트를 옮길 수 있다. `--config <json>`으로 모든 인자 기본값을 파일에서 줄 수 있고(예: [configs/example_config.json](../../configs/example_config.json)), 명시 CLI 플래그가 config보다 우선한다. 입력 경로(`--preprocessed_dir`·`--folds_csv`)는 기존대로 override 가능.
- **✅ 데이터 검증**: `load_preprocessed_data_with_folds`가 이제 **중복 ID·`range(6)` 밖 fold·NaN fold·누락 `.npy`** 를 명시적으로 경고(`[WARN]`)한다. 기본은 경고만(실제 로드되는 표본은 불변), `--strict_data`를 주면 동일 조건에서 `ValueError`로 중단한다. → 무음 누락/중복 덮어쓰기를 가시화.
- 🟢 잔존(미수정): 입력 **96×96×48·3채널 강결합**(patch_embed·`input_resolution`). 다른 크기는 형상 오류 — 아키텍처 변경이라 보존. 기본 입력 경로도 여전히 `/home/rintern07/...`(override는 가능).

## 12. 🟢 dynamo 오류 억제

파일 상단 `torch._dynamo.config.suppress_errors = True`. load-balance loss의 Python for-loop·boolean 인덱싱이 `torch.compile` 그래프 캡처와 충돌하는데, 이 설정이 컴파일 실패를 eager로 조용히 폴백시킨다 — 성능 저하가 숨겨질 수 있다.

## 13. 🟠 `shift_size` 미사용 — shifted-window 미구현

`MoHLGTBlock.__init__`이 `shift_size` 인자를 받아 `self.shift_size`에 저장하지만 `forward` 어디에서도 쓰지 않는다(`torch.roll`·윈도우 시프트·attention mask 로직 자체가 없음). 즉 Swin류 **shifted-window attention이 구현되지 않았고**, `shift_size`에 어떤 값을 넣어도 무효다(#2 `drop_path` no-op와 정확히 동형). NeoNet은 이 인자를 넘기지도 않아 항상 0이지만, 시프트로 윈도우 간 정보 교환을 켜려 해도 동작하지 않는다 → `torch.roll` 기반 시프트+마스킹을 구현해야 한다.

## 14. ✅ 재시작 시 CSV truncate (해결 2026-06-27)

(구) 체크포인트로 "계속 학습"해도 `neonet_training_metrics.csv`·`neonet_val_predictions.csv`를 무조건 `open(..., "w")`로 열어 이전 이력을 truncate(소실)했고, best-epoch 주석도 재시작 이후 구간만 반영했다.

- **✅ 해결**: 헤더 기록을 `if not (resuming and os.path.exists(csv_path))` 로 가드한다. **fresh run**(resuming=False)은 기존대로 `"w"`로 헤더를 쓰고(동작 불변), **resume**일 때만 헤더를 건너뛰어 append로 이력을 보존한다. 이로써 plot의 best-epoch도 전체 이력에서 계산된다.

## 15. ✅ plot 범례 정정 (해결 2026-06-27)

(구) `plot_metrics`의 load-balance 범례가 `"Load Balance Loss (×100)"`인데 실제 plot은 미스케일이라 100배 과대 표기였다.

- **✅ 해결**: 범례를 `"Load Balance Loss"`로 정정(실제 미스케일 plot과 일치). 모델·메트릭에는 무관(플롯 라벨만).

## 16. 🟢 재현성 한계

- DataLoader가 `worker_init_fn`·`generator`를 지정하지 않는다 → `num_workers>0`의 forked 워커에서 MONAI 증강 RNG가 시드에 고정되지 않을 수 있다(`set_determinism`만으로 fork 워커 결정성을 보장한다는 통념은 부정확).
- cuDNN 결정성 플래그(`torch.backends.cudnn.deterministic`/`benchmark`, `torch.use_deterministic_algorithms`)를 설정하지 않는다.
- `torch.cuda.manual_seed`(단일 device)만 쓰고 `manual_seed_all`은 쓰지 않는다.
- AMP는 이제 `--no-use_mixed_precision`으로 끌 수 있지만(✅ #5), 위 워커 시드·cuDNN 요인이 남아 `num_workers>0`에서 완전한 비트 단위 재현은 여전히 어렵다. 엄밀 재현이 필요하면 그 항목을 코드에서 보강해야 한다.

## 관련 문서

- [explanation/moh-lgt.md](moh-lgt.md) — #1·#3·#9의 메커니즘.
- [explanation/saf-fusion.md](saf-fusion.md) — #10.
- [reference/configuration.md](../reference/configuration.md) — #5·#11 설정.
