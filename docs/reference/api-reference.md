# API 레퍼런스 — 클래스·함수 명세

> **범위:** [MMA-Former.py](../../MMA-Former.py)의 모든 public 클래스·함수 시그니처와 입출력 계약을 조회용으로 정리한다. 설계 의도·근거는 explanation 문서에 있다.
> **대상:** 코드를 호출·확장하려는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-27. 코드 참조는 심볼명 기준(줄번호 없음).

## 1. 데이터

### `PreprocessedDataset(Dataset)`

`(file_paths, pni_labels, pids, transform=None, selected_channels=None)`. `__getitem__`는 `(image, label, pid)` 반환 — `image`는 `float32` 텐서 `(C,D,H,W)`(채널 선택·transform 적용 후), `label`은 `float32` 스칼라, `pid`는 str.

- ⚠️ `transform`을 주면 **MONAI dict-transform**이어야 한다 — `__getitem__`이 `transform({"input": image})["input"]`로 호출하므로 리터럴 키 `"input"`을 받고 돌려주는 변환만 동작한다. 평범한 텐서 콜러블을 넘기면 런타임 오류.

### `load_preprocessed_data_with_folds(preprocessed_dir, folds_csv_path, val_fold=0, strict=False)`

fold CSV를 읽어 train/val 경로·라벨·pid 6-튜플 반환: `(train_paths, train_labels, train_pids, val_paths, val_labels, val_pids)`. 컬럼 `ID·fold·PNI` 필수. 파일명 `patient_{id}.npy`/`patient{id}.npy` 양쪽 시도. 스키마는 [reference/data-model.md](data-model.md) §2.

- `strict`(기본 `False`): 중복 ID·`range(6)` 밖/NaN fold·누락 `.npy`를 발견하면 `False`는 `[WARN]`만(로드되는 표본 불변), `True`는 `ValueError`로 중단(✅ 2026-06-27). `--strict_data`로 연결.

## 2. 윈도우 파티셔닝

### `WindowPartitioner(max_cache_size=10)`

Swin 스타일 3D 윈도우 분할. 형상/permute 정보를 `(B,D,H,W,C,Wd,Wh,Ww)` 키로 LRU 캐싱. 모듈 레벨 싱글턴 `window_partitioner`로 전역 공유.

| 메서드 | 시그니처 | 반환 |
|---|---|---|
| `calculate_padding` | `(D,H,W,window_size)` | `(pad_d,pad_h,pad_w)` — 윈도우로 나눠떨어지게 하는 우측 패딩 |
| `partition` | `(x, window_size)` | `(windows, cached_info)` — `windows`는 `(num_windows·B, Wd·Wh·Ww, C)` |
| `reverse` | `(windows, cache_info, B,D,H,W)` | 윈도우를 `(B,D,H,W,C)`로 복원 |

- 래퍼 함수 `window_partition_optimized(x, window_size)`·`window_reverse_optimized(...)`는 싱글턴에 위임.
- ⚠️ `reverse`/`window_reverse_optimized`의 `B,D,H,W` 인자는 **사용되지 않는다** — 복원은 짝이 되는 `partition` 호출이 캐시에 남긴 형상(`cache_info`의 `reverse_shape`/`reverse_final`)만 쓴다. 따라서 반드시 그 `cache_info`를 만든 `partition`과 짝지어 호출해야 하며 임의의 `B,D,H,W`로 reshape할 수 없다.

## 3. MoH 어텐션

### `MoHWindowAttention3D(nn.Module)`

```text
(dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
 moh_efficiency=0.75, num_shared_heads=2)
```

Mixture-of-Heads 윈도우 어텐션. 헤드를 **shared(always-on) + routed(top-k 선택)**로 나눈다. 설계 상세는 [explanation/moh-lgt.md](../explanation/moh-lgt.md).

- **`forward(x, return_load_balance_loss=False)`** — `x`: `(B_, N, C)`.
  - `return_load_balance_loss=False` → `output` `(B_, N, C)`.
  - `True` → `(output, load_balance_loss)`. `load_balance_loss`는 `num_routed_heads>0`일 때만 비-0.
- 주요 서브모듈: `qkv` `Linear(dim,3·dim)`, `proj` `Linear(dim,dim)`, 그리고 `router`·`alpha_router`(**둘 다 `num_routed_heads>0`일 때만 생성** — 같은 `if` 블록). `router` = `Linear(dim,dim/4)→ReLU→Linear(dim/4,num_routed_heads)`, `alpha_router` = `Linear(dim,2)`.
- 🔴 `dim_proj`: `__init__`에서 `None`. MoH 경로에서 출력 마지막 차원이 `head_dim`이 되면 `forward` 중 lazy로 `Linear(head_dim,dim)` 생성. **옵티마이저 생성 이후 만들어져 학습되지 않음** — [explanation/known-issues.md](../explanation/known-issues.md) #1.

### `MoHLGTBlock(nn.Module)`

```text
(dim, input_resolution, num_heads, window_size_local=(3,3,3), window_size_global=(6,6,3),
 shift_size=0, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
 norm_layer=nn.LayerNorm, moh_efficiency=0.75, num_shared_heads=2)
```

Local-Global Transformer 블록. local·global 두 윈도우 크기로 병렬 MoH 어텐션을 수행하고 `fusion_proj` `Linear(2·dim, dim)`으로 융합. `forward(x, return_load_balance_loss=False)` — `x`: `(B,C,D,H,W)` → 동형 반환.

- 서브모듈: `norm1_local`·`norm1_global`(LayerNorm), `attn_local`·`attn_global`(MoHWindowAttention3D), `fusion_proj`, `norm2`, `mlp`(`Linear→GELU→Dropout→Linear→Dropout`, hidden = `dim·mlp_ratio`).
- 🟠 `shift_size`: `self.shift_size`에 저장만 되고 `forward`에서 **미사용**(shifted-window 미구현) — [explanation/known-issues.md](../explanation/known-issues.md) #13.
- 🟠 `drop_path`: `nn.Identity() if drop_path==0. else nn.Identity()` → 항상 no-op.
- 잔차: `x = shortcut + drop_path(x_fused)` 후 `x = x + drop_path(mlp(norm2(x)))`.

## 4. 융합

### `SAF(nn.Module)`

`(in_channels_a, in_channels_b, out_channels)`. Spatial Attention Fusion — 서로 다른 해상도/채널의 두 피처를 융합. `forward(feat_a, feat_b)` → `(B, out_channels, *feat_a.spatial)`. 단계(cross-attn → spatial → channel → gated fusion → residual)는 [explanation/saf-fusion.md](../explanation/saf-fusion.md).

- `common_dim = min(in_channels_a, in_channels_b)`. `feat_b`는 `feat_a` 공간 크기로 trilinear interpolate.
- `cross_attn` `MultiheadAttention(common_dim, heads=4, batch_first=True, dropout=0.1)` — **양방향 공유 모듈**.
- `residual_proj`: `in_channels_a != out_channels`일 때만 생성(기본 saf1/saf2는 `None`).

## 5. 모델

### `NeoNet(nn.Module)`

`(in_channels=3, moh_efficiency=0.75, num_shared_heads=2)`. `forward(x, return_load_balance_loss=False)` — `x`: `(B, in_channels, 96, 96, 48)` → `output` `(B,)` raw logit. `True`이면 `(output, total_load_balance_loss)`(3개 블록 합).

- 구성: `patch_embed` → `moh_lgt_block1` → `downsample1` → `moh_lgt_block2` → `saf1(x2,x1)` → `downsample2` → `moh_lgt_block3` → `saf2(x3,x2_fused)` → `global_pool` → `classifier`. 단계별 형상 [reference/data-model.md](data-model.md) §3.

## 6. 메트릭·로깅

| 함수 | 시그니처 | 반환/효과 |
|---|---|---|
| `compute_metrics` | `(preds, trues)` | `(acc, f1, dice, auc)`. `preds`는 연속 확률, 0.5 임계로 이진화. 🟢 `dice`는 이진 분류에서 `f1`과 수학적으로 동일. `auc`는 `try/except`로 실패 시 `nan` |
| `plot_metrics` | `(metrics_df, save_dir, timestamp, fold, lr, seed)` | 매 10 epoch 6분할 PNG 저장 |
| `save_best_scores_to_csv` | `(metrics_df, fold, lr, seed, timestamp, best_scores_dir, args)` | best(val_loss 최저) epoch 스냅샷 CSV 누적 |
| `combined_loss_fn` | `(logits, labels, load_balance_loss, weight=0.005)` | `(total, bce, lb)`. `total = BCEWithLogitsLoss(pos_weight=1.5) + weight·lb`. ⚠️ `__main__` 내부 중첩 함수 — import 불가, `device`를 closure로 캡처 |

## 관련 문서

- [explanation/architecture.md](../explanation/architecture.md) — 위 컴포넌트가 어떻게 조립되는가.
- [explanation/known-issues.md](../explanation/known-issues.md) — 시그니처만으로는 안 보이는 함정.
