# How-to — MoH 하이퍼파라미터 조정

> **범위:** MoH 어텐션 거동을 바꾸는 세 인자 `--moh_efficiency`·`--num_shared_heads`·`--load_balance_weight`의 의미와 조정 효과. 메커니즘 자체는 [explanation/moh-lgt.md](../explanation/moh-lgt.md).
> **대상:** MoH 설정을 실험하는 연구자.
> **상태:** 구현 반영 — 기준일 2026-06-27.

## 1. 세 인자 한눈에

| 인자 | 기본 | 영향 |
|---|---|---|
| `--num_shared_heads` | 2 | always-on 공유 헤드 수. 라우팅에서 제외되는 헤드 |
| `--moh_efficiency` | 0.75 | routed 헤드 중 top-k 선택 비율 |
| `--load_balance_weight` | 0.005 | 라우터 균형 보조손실의 총손실 내 가중 |

블록별 헤드 수는 6/8/12로 고정(CLI 노출 안 됨). 파생값 계산은 [reference/configuration.md](../reference/configuration.md) §2.

## 2. `num_shared_heads`

```text
num_routed_heads = num_heads - num_shared_heads
```

- **올리면**: 라우팅 대상 헤드가 줄어 MoH 특화 효과↓, 표준 어텐션에 가까워짐.
- `num_shared_heads ≥ num_heads`로 두면 `num_routed_heads=0` → 라우팅이 꺼지고 **dense 경로**(헤드 concat)로 전환. 단 이 경로는 거의 테스트되지 않은 dead path이며 MoH 경로와 차원 처리가 다르다([known-issues.md](../explanation/known-issues.md) #3).
- block1 헤드가 6이므로 공유를 6 이상으로 두면 block1은 dense, 다른 블록은 다르게 동작할 수 있다 — 블록마다 `min(num_shared_heads, num_heads)`로 클램프됨에 유의.

## 3. `moh_efficiency`

```text
top_k = max(1, int(num_routed_heads · moh_efficiency))
```

- **1.0에 가까울수록** routed 헤드를 거의 다 사용(MoH의 sparsity 이점↓, 연산↑).
- **낮출수록** 윈도우당 더 적은 헤드만 활성(sparsity↑). 단 `max(1, …)`로 최소 1개는 보장.
- 예(기본 헤드, 공유 2): block3 `num_routed_heads=10` → eff 0.75면 top_k=7, eff 0.5면 5, eff 0.3이면 3.

## 4. `load_balance_weight`

라우터가 소수 헤드에 쏠리는 것을 막는 보조손실 가중치.

- **0으로 두면**: 균형 유도 없음 → 라우터가 헤드 일부만 쓰는 collapse 위험.
- **너무 키우면**: 주손실(BCE) 학습을 방해. 기본 0.005는 작게 잡힌 값.
- ⚠️ 이 항은 학습뿐 아니라 **검증 손실에도 포함**되어 best-model 선택 기준을 흔든다([known-issues.md](../explanation/known-issues.md) #8). MoH 가중을 크게 실험할 때 모델 선택 편향에 주의.

## 5. 권장 실험 순서

1. 기본값으로 baseline 확보(단, 🔴 `dim_proj` 미학습 이슈를 먼저 인지 — [known-issues.md](../explanation/known-issues.md) #1. 이 버그가 남아 있으면 MoH 튜닝 효과 해석이 오염된다).
2. `load_balance_weight`를 0 / 0.005 / 0.02로 스윕해 라우터 collapse 여부 관찰(`load_balance_loss` 컬럼 추이).
3. `moh_efficiency`를 0.5–1.0로 스윕해 sparsity↔성능 트레이드오프 확인.
4. `num_shared_heads`는 1–3 좁은 범위에서만(그 이상은 dense 경로 전환 주의).

## 관련 문서

- [explanation/moh-lgt.md](../explanation/moh-lgt.md) — top_k·alpha 게이트·load balance 수식.
- [explanation/known-issues.md](../explanation/known-issues.md) — 튜닝 결과 해석 전 필독.
