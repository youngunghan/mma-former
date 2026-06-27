# 아키텍처 — NeoNet 엔드투엔드

> **범위:** [MMA-Former.py](../../MMA-Former.py) `NeoNet`의 전체 데이터 흐름과 설계 의도를 설명한다. MoH 어텐션·SAF 내부는 형제 문서([moh-lgt.md](moh-lgt.md)·[saf-fusion.md](saf-fusion.md))에 위임한다.
> **대상:** 모델 설계를 이해하려는 연구자.
> **상태:** 구현 반영 — 코드 대조 기준일 2026-06-27.

## 1. 한눈에

NeoNet은 **3D 의료영상 볼륨 → PNI 이진 라벨**을 예측하는 계층형 트랜스포머다. Swin류 윈도우 어텐션을 3D로 확장하되, 각 단계의 트랜스포머 블록을 **MoH-LGT**(Mixture-of-Heads Local-Global Transformer)로 구성하고, 인접 단계 간에 **SAF**(Spatial Attention Fusion)로 멀티스케일 특징을 결합한다.

- **계보:** Med-Former 백본의 LGT 모듈을 MoH 어텐션으로 교체한 변종(상위 [README.md](../../README.md)).
- **출력:** 단일 raw logit. 손실은 `BCEWithLogitsLoss`(+ MoH load-balance 보조손실).

```text
입력 볼륨 (B,3,96,96,48)
      │
      ▼  patch_embed: Conv3d(3→48, k4 s4)        4배 다운샘플
 (B,48,24,24,12)
      │
      ▼  ┌─ Stage 1 ──────────────────────────┐
      │  │ MoHLGTBlock dim48 heads6           │ → x1
      │  │  local(3,3,3) ∥ global(6,6,6)      │
      │  └────────────────────────────────────┘
      ▼  downsample1: Conv3d(48→96, k2 s2)
 (B,96,12,12,6)
      │
      ▼  ┌─ Stage 2 ──────────────────────────┐
      │  │ MoHLGTBlock dim96 heads8           │ → x2
      │  └────────────────────────────────────┘
      │       │
      │       ▼  SAF1(96,48,96): fuse(x2, x1)   ← x1을 (12,12,6)로 다운샘플 후 결합
 (B,96,12,12,6)  x2_fused
      │
      ▼  downsample2: Conv3d(96→192, k2 s2)
 (B,192,6,6,3)
      │
      ▼  ┌─ Stage 3 ──────────────────────────┐
      │  │ MoHLGTBlock dim192 heads12         │ → x3
      │  │  local(3,3,3) ∥ global(6,6,3)      │
      │  └────────────────────────────────────┘
      │       │
      │       ▼  SAF2(192,96,192): fuse(x3, x2_fused)
 (B,192,6,6,3)  x3_fused
      │
      ▼  global_pool AdaptiveAvgPool3d(1) → flatten   (B,192)
      ▼  classifier: 192→64→ReLU→Dropout(0.15)→64→1   (B,)  raw logit
```

## 2. 단계별 설계 의도

### 2.1 Patch embedding

`Conv3d(in, 48, kernel=4, stride=4)`로 입력을 한 번에 4배 다운샘플하며 토큰화한다. 결과 `(B,48,24,24,12)`는 stage1의 `input_resolution`과 일치하도록 맞춰져 있다(전처리 96×96×48 가정에 강결합).

### 2.2 계층 다운샘플

각 stage 사이에 `Conv3d(k2,s2)`로 공간을 절반, 채널을 2배(48→96→192)로 만든다. Swin의 patch merging을 conv로 대체한 형태다. 채널이 깊어질수록 헤드 수도 6→8→12로 증가한다.

### 2.3 MoH-LGT 블록 (각 stage)

블록은 입력을 두 윈도우 크기로 **병렬** 처리한다: local(작은 수용야, `(3,3,3)`)과 global(큰 수용야, `(6,6,6)` 또는 `(6,6,3)`). 각 분기는 독립 LayerNorm + MoH 어텐션을 거친 뒤 `fusion_proj`(`Linear(2C→C)`)로 합쳐진다. 이렇게 **국소 디테일과 광역 맥락을 한 블록에서 동시에** 본다. 내부 라우팅·load balance는 [moh-lgt.md](moh-lgt.md).

### 2.4 SAF 멀티스케일 융합

- **SAF1**: stage2 출력(저해상·고채널)을 stage1 출력(고해상·저채널)과 결합.
- **SAF2**: stage3 출력을 SAF1 결과와 결합.

각 SAF는 cross-attention으로 두 스케일을 상호 참조시킨 뒤 spatial/channel attention과 gated fusion으로 합친다([saf-fusion.md](saf-fusion.md)). 주의: SAF는 항상 `feat_b`(고해상 특징)를 `feat_a`(저해상) 크기로 **다운샘플**한 뒤 융합하므로, 고해상 정보가 손실될 수 있다(설계 트레이드오프, [known-issues.md](known-issues.md) 참조).

### 2.5 분류 헤드

최종 `(B,192,6,6,3)`을 global average pooling으로 `(B,192)`로 줄이고 `192→64→1` MLP로 단일 logit을 낸다. 시그모이드는 forward에 없고 손실/메트릭 단계에서만 적용된다.

## 3. 학습 신호

- **주손실:** `BCEWithLogitsLoss(pos_weight=1.5)` — 양성(PNI=1) 가중을 1.5배.
- **보조손실:** 각 MoH 어텐션의 load-balance loss 합을 `load_balance_weight`(0.005)로 가중해 더함. 라우터가 특정 헤드에만 쏠리지 않게 유도한다([moh-lgt.md](moh-lgt.md) §load balance).
- 최적화 AdamW(lr 8e-5, wd 0.01), AMP 기본 on, 200 epoch·patience 30(val_loss 최저 기준)로 early stop.

## 4. 강결합·전제

- 입력 공간 크기 **96×96×48 고정** 전제(patch_embed·각 `input_resolution`이 이 값에 맞춰 하드코딩). 다른 크기를 넣으면 윈도우 분할이 어긋나거나 형상 불일치가 난다.
- 채널 수 3 가정(또는 `--selected_channels`로 명시). 채널 의미는 전처리 코드 소관(이 저장소 밖).

## 관련 문서

- [explanation/moh-lgt.md](moh-lgt.md) — 블록 내부 MoH 메커니즘.
- [explanation/saf-fusion.md](saf-fusion.md) — SAF 융합 단계.
- [reference/data-model.md](../reference/data-model.md) — 단계별 텐서 형상 표.
- [explanation/known-issues.md](known-issues.md) — 이 아키텍처의 알려진 함정.
