# MoH-LGT — Mixture-of-Heads 윈도우 어텐션과 Local-Global 블록

> **범위:** [MMA-Former.py](../../MMA-Former.py) `MoHWindowAttention3D`와 `MoHLGTBlock`의 동작을 설명한다. 이 변종이 프로젝트명 *MMA-Former*의 핵심(Med-Former LGT → MoH)이다. 윈도우 분할 헬퍼(`WindowPartitioner`)도 다룬다.
> **대상:** 어텐션 메커니즘을 검토·수정하려는 연구자.
> **상태:** 구현 반영 — 코드 대조 기준일 2026-06-27. 일부 구현 특이점은 [known-issues.md](known-issues.md)로 분리.

## 1. 배경 — 왜 MoH인가

표준 멀티헤드 어텐션은 모든 헤드를 매 토큰에 동일하게 적용한다. **MoH(Mixture-of-Heads)**는 헤드를 전문가처럼 취급해, 일부는 항상 켜두고(shared) 나머지는 입력에 따라 **top-k로 선택**(routed)함으로써 헤드별 특화와 연산 절감을 노린다. 본 구현은 이를 3D 윈도우 어텐션에 얹었다.

> ⚠️ **구현 주의:** 본 코드의 MoH는 표준 MoH 논문과 두 가지가 다르다 — (a) 라우팅이 **토큰 단위가 아니라 윈도우 단위**, (b) 헤드 출력을 concat이 아니라 **합산**해 `head_dim`으로 붕괴시킨 뒤 `dim_proj`로 복원. 이로 인한 부작용은 [known-issues.md](known-issues.md).

## 2. 윈도우 파티셔닝 (`WindowPartitioner`)

Swin의 3D 확장. 입력 `(B,D,H,W,C)`를 윈도우 `(Wd,Wh,Ww)`로 잘라 `(num_windows·B, Wd·Wh·Ww, C)` 토큰 시퀀스를 만든다.

- `calculate_padding`은 각 축을 윈도우로 나눠떨어지게 우측 패딩량을 구한다. **기본 블록 설정에서는 모든 해상도가 윈도우로 나눠떨어져 패딩이 0**이다:

  | 블록 | 해상도 | local (3,3,3) | global |
  |---|---|---|---|
  | 1 | (24,24,12) | 8,8,4 ✓ | (6,6,6) → 4,4,2 ✓ |
  | 2 | (12,12,6) | 4,4,2 ✓ | (6,6,6) → 2,2,1 ✓ |
  | 3 | (6,6,3) | 2,2,1 ✓ | (6,6,3) → 1,1,1 ✓ |

- `view → permute → contiguous → view`로 분할하고 역과정으로 복원한다. 형상/permute 튜플을 `(B,D,H,W,C,Wd,Wh,Ww)` 키로 LRU 캐싱하지만 캐시는 메타데이터(튜플)만 담아 실연산 절감은 작다. 전역 싱글턴 `window_partitioner`로 모든 블록이 공유한다.

## 3. MoH 윈도우 어텐션 (`MoHWindowAttention3D`)

입력 윈도우 토큰 `(B_, N, C)`. `qkv = Linear(dim, 3·dim)` 후 `q,k,v: (B_, num_heads, N, head_dim)`. 헤드를 두 그룹으로 나눈다:

- `num_shared_heads`(기본 2): 항상 켜지는 공유 헤드.
- `num_routed_heads = num_heads - num_shared_heads`: 라우터가 top-k 선택.
- `top_k = max(1, int(num_routed_heads · moh_efficiency))`(기본 `moh_efficiency=0.75`).

### 3.1 라우터 — 윈도우 단위 결정

```text
window_repr = x.mean(dim=1)            # (B_, C)  ← 윈도우 내 토큰 평균
router_logits = router(window_repr)    # (B_, num_routed_heads)
router_probs  = softmax(router_logits)
top_k_probs, selected_routed_indices = topk(router_probs, top_k)   # (B_, top_k)
top_k_weights = top_k_probs / (sum + 1e-8)                         # 정규화 가중
```

`router`는 `Linear(dim, dim/4)→ReLU→Linear(dim/4, num_routed_heads)`. 라우팅 단위가 **윈도우 1개당 1회**이므로, 같은 윈도우의 모든 토큰은 동일한 routed 헤드 집합을 쓴다.

### 3.2 alpha 게이트 — shared vs routed 비중

```text
alpha = softmax(alpha_router(window_repr))   # (B_, 2)
alpha_shared, alpha_routed = alpha[:,0:1], alpha[:,1:2]
```

`alpha_router = Linear(dim, 2)`. 두 경로의 기여 비율을 윈도우별로 학습한다.

### 3.3 shared 경로

```text
shared_q/k/v = q/k/v[:, :num_shared_heads]          # (B_, S, N, head_dim)
shared_attn  = softmax(shared_q @ shared_kᵀ)        # (B_, S, N, N)
shared_output = (shared_attn @ shared_v).sum(dim=1) # (B_, N, head_dim)  ← 헤드 합산
shared_output *= alpha_shared
```

### 3.4 routed 경로

advanced indexing으로 윈도우별 선택 헤드를 모은다:

```text
routed_head_indices = selected_routed_indices + num_shared_heads     # 공유 헤드 뒤쪽
routed_q/k/v = q/k/v[batch_indices, routed_head_indices]             # (B_, top_k, N, head_dim)
routed_attn  = softmax(routed_q @ routed_kᵀ)
routed_output = (routed_attn @ routed_v)
routed_output *= (top_k_weights · alpha_routed)                      # 선택 가중 × 게이트
routed_output = routed_output.sum(dim=1)                             # (B_, N, head_dim)  ← 헤드 합산
```

### 3.5 결합과 차원 복원

```text
combined_output = shared_output + routed_output    # (B_, N, head_dim)   ← C 아님!
if combined_output.size(-1) != C:                  # MoH 경로에서 항상 참
    dim_proj = Linear(head_dim, C)  (lazy 생성)     # 🔴 학습 안 됨
    combined_output = dim_proj(combined_output)
output = proj(combined_output)                     # (B_, N, C)
```

- 헤드를 **합산**하므로 마지막 차원이 `head_dim`으로 붕괴되고, `dim_proj`가 `head_dim→C`로 되돌린다. 이 `dim_proj`는 `forward` 중 lazy 생성되어 **옵티마이저에 포함되지 않고 학습되지 않는다** — 본 구현의 가장 큰 결함([known-issues.md](known-issues.md) #1).
- 참고: `num_routed_heads==0`인 dead 경로(기본 설정에선 미사용)는 헤드를 **concat**해 `C`를 유지한다 → 두 경로가 차원적으로 불일치.

### 3.6 Load-balance loss

`return_load_balance_loss=True`이고 `num_routed_heads>0`일 때만:

```text
P_i = router_probs.mean(dim=0)                       # 헤드 i의 평균 라우팅 확률
f_i = fraction of top-k slots assigned to head i     # Python for-loop로 계산
load_balance_loss = Σ_i (P_i · f_i) · num_routed_heads
```

Switch-Transformer류 보조손실로, 특정 헤드 독점을 억제한다. `f_i`를 파이썬 루프 + boolean 비교로 구해 `torch.compile`(dynamo)와 충돌하며, 그래서 파일 상단에 `torch._dynamo.config.suppress_errors = True`가 설정돼 있다(컴파일 실패를 조용히 무시).

## 4. LGT 블록 (`MoHLGTBlock`)

`(B,C,D,H,W)`를 받아 `(B,D,H,W,C)`로 permute한 뒤 두 분기를 **병렬** 실행한다:

```text
shortcut = x
x_local  = norm1_local(x)   → (pad) → window_partition(local)  → attn_local  → reverse → (unpad)
x_global = norm1_global(x)  → (pad) → window_partition(global) → attn_global → reverse → (unpad)
x_fused  = fusion_proj( concat([x_local, x_global], dim=-1) )    # Linear(2C→C)
x = shortcut + drop_path(x_fused)                                # 🟠 drop_path는 no-op
x = x + drop_path(mlp(norm2(x)))                                 # MLP: C→2C→GELU→C
permute back → (B,C,D,H,W)
```

- **local ∥ global**: 같은 입력을 작은/큰 윈도우로 동시에 보고 융합 — 국소 디테일 + 광역 맥락. 이것이 "Local-Global Transformer"의 핵심.
- `mlp_ratio=2`로 표준 4보다 얕다(3D 메모리 절약 의도로 보임).
- 🟠 `drop_path`는 ternary 양 분기가 모두 `nn.Identity()`라 stochastic depth가 실제로 적용되지 않는다.

## 관련 문서

- [explanation/architecture.md](architecture.md) — 이 블록들이 NeoNet에 어떻게 배치되는가.
- [explanation/known-issues.md](known-issues.md) — dim_proj 미학습·drop_path no-op·라우팅 단위 등.
- [reference/api-reference.md](../reference/api-reference.md) — 시그니처.
