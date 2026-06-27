# SAF — Spatial Attention Fusion

> **범위:** [MMA-Former.py](../../MMA-Former.py) `SAF` 모듈의 멀티스케일 융합 단계를 설명한다. 어디에 배치되는지는 [architecture.md](architecture.md), MoH 어텐션은 [moh-lgt.md](moh-lgt.md).
> **대상:** 융합 모듈을 검토·수정하려는 연구자.
> **상태:** 구현 반영 — 코드 대조 기준일 2026-06-27. 클래스 docstring에 "Fixed version with Cross Attention → Spatial Attention"으로 표기된 개정본.

## 1. 역할

SAF는 **해상도·채널이 다른 두 특징 맵**(깊은 stage의 저해상·고채널 `feat_a`와 얕은 stage의 고해상·저채널 `feat_b`)을 하나로 합친다. NeoNet은 SAF1(stage2 ⊕ stage1)·SAF2(stage3 ⊕ SAF1결과) 두 번 사용한다.

호출 규약: `forward(feat_a, feat_b)`에서 출력 공간 크기는 **`feat_a` 기준**이고, `feat_b`는 거기에 맞춰 trilinear interpolate된다. NeoNet 호출(`saf1(x2, x1)`, `saf2(x3, x2_fused)`)에서 `feat_a`가 저해상이므로 **고해상 `feat_b`가 다운샘플**된다.

## 2. 단계별 흐름

```text
feat_a (in_a ch, 저해상)         feat_b (in_b ch, 고해상)
   │                                │ interpolate → feat_a 공간크기 (trilinear)
   ▼                                ▼
 proj_a: Conv1³(in_a→common)+BN   proj_b: Conv1³(in_b→common)+BN     common = min(in_a,in_b)
   │                                │
   └────────────┐      ┌────────────┘
                ▼      ▼
        [1] Cross-Attention (양방향, 공유 모듈)
            attn_a = cross_attn(a, b, b)      # a가 b를 질의
            attn_b = cross_attn(b, a, a)      # b가 a를 질의
            feat_a_cross = proj_a + attn_a    # 잔차
            feat_b_cross = proj_b + attn_b
                │      │
                ▼      ▼
        [2] Spatial Attention (분기별)
            w_a = spatial_attn_a(feat_a_cross)   # Conv→BN→ReLU→Conv→Sigmoid → (B,1,D,H,W)
            feat_a_spatial = feat_a_cross * (1 + w_a)   # residual gating
            (feat_b도 동일)
                │      │
                └──┬───┘ concat (dim=1) → (B, 2·common, ...)
                   ▼
        [3] Channel Attention (SE류)
            cw = channel_attn(fused)   # GAP→Conv→ReLU→Conv→Sigmoid
            fused = fused * cw
                   ▼
        [4] Gated Fusion
            g = gate(fused)            # Conv→Sigmoid
            fused = fused * g
                   ▼
        [5] final_proj: Conv1³(2·common→out)+BN+ReLU
                   ▼
        [6] Residual: output + identity
            identity = feat_a  (또는 residual_proj(feat_a) if in_a≠out)
```

## 3. 각 단계 의도

### 3.1 공통 차원 투영

두 입력을 `common_dim = min(in_a, in_b)` 채널로 1×1×1 conv 투영(+BN)해 cross-attention이 가능한 동일 차원으로 맞춘다.

### 3.2 Cross-attention (FIRST)

`feat_a`가 `feat_b`를, `feat_b`가 `feat_a`를 상호 질의해 스케일 간 대응을 학습한다. 잔차로 원 투영 특징에 더한다.

- **공유 모듈 주의:** 두 방향(`a→b`, `b→a`)이 **같은 `self.cross_attn` 인스턴스**를 쓴다(파라미터 공유). 별도 모듈을 둘 수도 있었으나 현재는 한 개를 재사용한다([known-issues.md](known-issues.md) 참조).
- `MultiheadAttention(common_dim, heads=4, batch_first=True, dropout=0.1)`. 시퀀스 길이는 `D·H·W`(예: SAF1에서 12·12·6 = 864).

### 3.3 Spatial attention (SECOND)

각 분기에서 `(B,1,D,H,W)` 공간 가중치를 만들어 `feat * (1 + weight)`로 곱한다. `+1` 잔차 게이팅이라 가중치가 0이어도 신호가 죽지 않는다.

### 3.4 Channel attention + Gated fusion

concat된 `2·common` 채널에 SE류 채널 어텐션과 sigmoid 게이트를 차례로 적용해 채널·위치별 중요도를 재조정한다.

### 3.5 최종 투영 + 잔차

`final_proj`로 `out_channels`로 사상하고 `feat_a`(필요 시 `residual_proj` 통과)를 더한다. 기본 NeoNet의 saf1(96→96)·saf2(192→192)는 `in_a==out`이라 `residual_proj=None`이며 `feat_a`가 그대로 더해진다.

## 4. 설계 트레이드오프

- **고해상 정보 다운샘플:** `feat_b`(얕은 stage의 디테일 풍부한 고해상 특징)를 `feat_a` 크기로 줄인 뒤 융합하므로, 세밀한 공간 정보가 일부 손실된다. U-Net류 skip이 보통 고해상 쪽에서 융합하는 것과 반대 방향이다 — 의도된 선택인지 확인 필요([known-issues.md](known-issues.md)).
- 모듈 깊이가 큼(cross-attn + 4종 attention/gate). 작은 데이터셋(168 규모)에서 과적합·파라미터 효율 관점의 검토 여지.

## 관련 문서

- [explanation/architecture.md](architecture.md) — SAF의 배치.
- [reference/api-reference.md](../reference/api-reference.md) `SAF` — 시그니처.
- [explanation/known-issues.md](known-issues.md) — 공유 cross_attn·다운샘플 트레이드오프.
