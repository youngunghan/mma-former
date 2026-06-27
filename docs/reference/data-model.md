# 데이터 모델 — 입력 포맷·fold 스키마·텐서 형상

> **범위:** 학습 입력(`.npy` 볼륨, fold CSV)의 온디스크 스키마와 [MMA-Former.py](../../MMA-Former.py) `NeoNet` 내부를 흐르는 텐서 형상을 다룬다. 전처리 파이프라인(원본 DICOM/NIfTI → cropped `.npy`)은 이 저장소 밖이라 범위 밖.
> **대상:** 모델·데이터 담당 개발자.
> **상태:** 구현 반영 — 코드 대조 검증 기준일 2026-06-27.

## 1. 전처리 볼륨 (`.npy`)

[MMA-Former.py](../../MMA-Former.py) `PreprocessedDataset.__getitem__`는 환자당 한 개의 `.npy` 파일을 `np.load`로 읽어 `torch.float32` 텐서로 변환한다.

| 항목 | 값 | 비고 |
|---|---|---|
| dtype | `float32`로 캐스팅 | 원본 dtype 무관, `.float()` 강제 |
| 형상 | `(C, D, H, W)` | 채널 우선. 기본 `C=3` |
| 기본 공간 크기 | `96 × 96 × 48` | 디렉터리명 `preprocessed_all_96x96x48_FIXED` 규약 |
| 채널 선택 | `selected_channels`로 `image[selected_channels]` 슬라이싱 | `--selected_channels "0,1,2"` 형태 |
| 파일명 규약 | `patient_{ID}.npy` **또는** `patient{ID}.npy` | 둘 다 시도, 둘 다 없으면 해당 환자 skip |

- 채널이 의미하는 모달리티(예: CT 윈도우·마스크 채널)는 코드에 명시되어 있지 않다 → **확인 필요**(전처리 코드 소관).
- 라벨은 fold CSV의 `PNI` 컬럼에서 오며 볼륨 파일에는 들어 있지 않다.

## 2. Fold CSV 스키마

[MMA-Former.py](../../MMA-Former.py) `load_preprocessed_data_with_folds`가 읽는다. `--folds_csv` 기본값은 `new_168_fold.csv`(168 환자 규모를 시사).

| 컬럼 | 타입 | 용도 |
|---|---|---|
| `ID` | str로 캐스팅 | 환자 식별자 → `patient_{ID}.npy` 매칭 키 |
| `fold` | int | 0–5 중 하나. 교차검증 분할 |
| `PNI` | 0/1 | 이진 라벨(타깃) |

- 세 컬럼이 모두 없으면 `ValueError`로 중단한다(컬럼명 대문자 `ID`·`PNI` 정확히 일치 필요).
- **분할 규칙:** validation = `fold == val_fold`인 행, training = `fold ∈ {0,1,2,3,4,5} \ {val_fold}`. 즉 하드코딩된 6-fold다.
- `fold` 값이 `range(6)` 밖이거나 NaN인 행, 파일이 없는 행, 중복 ID는 train/val에서 제외되거나 마지막 행으로 덮어써진다. **이제 모두 `[WARN]`으로 표시**되며 `--strict_data`면 `ValueError`로 중단한다(✅ 2026-06-27, 이전엔 무음). 기본(비-strict)에서 실제 로드되는 표본은 이전과 동일.

## 3. 단계별 텐서 형상 (기본 설정)

입력 `(B, 3, 96, 96, 48)` 기준. 채널·공간 변화를 단계별로 정리한다. 출처: `NeoNet.__init__` / `NeoNet.forward`.

```text
입력                      (B,   3, 96, 96, 48)
└─ patch_embed Conv3d k4s4 (B,  48, 24, 24, 12)   ← stage1 input_resolution (24,24,12)
   └─ moh_lgt_block1       (B,  48, 24, 24, 12)   heads=6, head_dim=8,  local(3,3,3) global(6,6,6)   → x1
      └─ downsample1 k2s2  (B,  96, 12, 12,  6)
         └─ moh_lgt_block2 (B,  96, 12, 12,  6)   heads=8, head_dim=12, local(3,3,3) global(6,6,6)   → x2
            └─ saf1(x2,x1) (B,  96, 12, 12,  6)   SAF(96,48,96): x1을 (12,12,6)로 trilinear 다운샘플 후 융합 → x2_fused
               └─ downsample2 k2s2 (B, 192, 6, 6, 3)
                  └─ moh_lgt_block3 (B, 192, 6, 6, 3)  heads=12, head_dim=16, local(3,3,3) global(6,6,3) → x3
                     └─ saf2(x3,x2_fused) (B, 192, 6, 6, 3)  SAF(192,96,192) → x3_fused
                        └─ global_pool AdaptiveAvgPool3d(1).flatten(1)  (B, 192)
                           └─ classifier 192→64→1, squeeze(1)           (B,)   ← raw logit
```

- 모든 `input_resolution`은 local(3,3,3)·global 윈도우로 정확히 나눠떨어져 **기본 설정에서 패딩 0**(상세: [explanation/moh-lgt.md](../explanation/moh-lgt.md) §윈도우 파티셔닝).
- head_dim = dim ÷ num_heads: block1 = 48/6 = 8, block2 = 96/8 = 12, block3 = 192/12 = 16.
- 출력은 **raw logit**(시그모이드 미적용). 손실은 `BCEWithLogitsLoss`, 추론 확률은 메트릭 계산 시 `torch.sigmoid`로만 구한다.

## 4. MoH 윈도우 어텐션 내부 형상

`MoHWindowAttention3D.forward` 입력은 윈도우 토큰 `(B_, N, C)`. `B_` = 배치 × 윈도우 수, `N` = 윈도우 부피(예: local 3·3·3 = 27).

| 텐서 | 형상 | 설명 |
|---|---|---|
| `qkv` 분해 후 q,k,v | `(B_, num_heads, N, head_dim)` | |
| `window_repr` | `(B_, dim)` | 토큰 평균(`x.mean(dim=1)`) → **윈도우 단위** 라우팅 입력 |
| `router_probs` | `(B_, num_routed_heads)` | softmax |
| `selected_routed_indices` | `(B_, top_k)` | top-k 선택 헤드 인덱스 |
| `shared_output` | `(B_, N, head_dim)` | shared 헤드 출력 **합산**(concat 아님) |
| `routed_output` | `(B_, N, head_dim)` | routed 헤드 출력 **합산** |
| `combined_output` | `(B_, N, head_dim)` | → 🔴 `dim_proj`(lazy)로 `head_dim→C` 복원 |
| `output` | `(B_, N, C)` | `proj` 후 |

- 🔴 MoH 경로에서 출력 마지막 차원이 `C`가 아니라 `head_dim`이 되어 `dim_proj`가 **항상** 작동한다. 이 `dim_proj`는 옵티마이저 생성 이후 lazy 생성되어 **학습되지 않는다**(상세: [explanation/known-issues.md](../explanation/known-issues.md)).

## 관련 문서

- [reference/configuration.md](configuration.md) — 위 형상을 결정하는 하이퍼파라미터.
- [explanation/architecture.md](../explanation/architecture.md) — 데이터 흐름의 설계 의도.
