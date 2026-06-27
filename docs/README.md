# MMA-Former (NeoNet) 문서

3D 의료영상 **PNI(Perineural Invasion, 신경주위 침범) 이진 분류**를 위한 연구용 모델 `NeoNet`의 문서 허브.
Med-Former 계열 백본의 **LGT(Local-Global Transformer) 모듈을 MoH(Mixture-of-Heads) 어텐션으로 교체**한 변종이다.

> ⚠️ 상위 `README.md` 명시: *"이전 서버 데이터 날아감 이슈로 최종버전이 아닙니다. Med-Former의 LGT 모듈에 MoH 버전이 구현된 버전입니다."* — **연구 진행 중(WIP) 코드**이며 최종본이 아니다. 본 문서는 단일 파일 [MMA-Former.py](../MMA-Former.py)를 코드 대조로 정리한 기록이다.
> **2026-06-27 안전 수정 반영:** 학습 결과/아키텍처를 바꾸지 않는 견고성·재현성 항목(resume·AMP 토글·출력 경로·데이터 검증·AUC·CSV·plot 라벨)을 코드에서 수정했다. 모델 동작을 바꾸는 결함(🔴 `dim_proj` 미학습 등)은 **의도적으로 보존**(기존 결과 호환). 상세 [explanation/known-issues.md](explanation/known-issues.md).

- 코드: [MMA-Former.py](../MMA-Former.py) — 데이터셋·모델·학습 루프가 모두 들어 있는 단일 스크립트.
- 상위 저장소: `SNU-MED-AI-for-Paper-publication`(MICCAI·ISBI·NeurIPS·AAAI·IEEE 투고 대상 연구 모노레포). 모델 클래스명 `NeoNet`은 상위 README의 `Ours/NeoNet` 항목과 대응한다.

## 문서 목록 (Diátaxis)

### Tutorials — 처음 따라하기

| 문서 | 설명 |
|---|---|
| [tutorials/quickstart.md](tutorials/quickstart.md) | 의존성 설치 → 전처리 데이터·fold CSV 준비 → 단일 fold 학습 실행 → 산출물 확인 |

### How-to — 목표 지향 가이드

| 문서 | 설명 |
|---|---|
| [how-to/run-training.md](how-to/run-training.md) | 6-fold 교차검증 실행, 채널 선택, 체크포인트 이어받기, 산출 CSV/플롯 해석 |
| [how-to/tune-moh.md](how-to/tune-moh.md) | MoH 하이퍼파라미터(`moh_efficiency`·`num_shared_heads`·`load_balance_weight`) 조정과 영향 |

### Reference — 조회용 명세

| 문서 | 설명 |
|---|---|
| [reference/api-reference.md](reference/api-reference.md) | 전 클래스·함수 시그니처(NeoNet·MoHWindowAttention3D·MoHLGTBlock·SAF·WindowPartitioner·Dataset·metrics) |
| [reference/configuration.md](reference/configuration.md) | CLI 인자 표 + 코드 내 하드코딩 하이퍼파라미터 + 경로 |
| [reference/data-model.md](reference/data-model.md) | `.npy` 볼륨 포맷, fold CSV 스키마, 네트워크 단계별 텐서 형상 |

### Explanation — 깊은 설명

| 문서 | 설명 |
|---|---|
| [explanation/architecture.md](explanation/architecture.md) | NeoNet 엔드투엔드 파이프라인(patch embed → 3× MoH-LGT + downsample → 2× SAF → head) |
| [explanation/moh-lgt.md](explanation/moh-lgt.md) | MoH 윈도우 어텐션(shared/routed 헤드·top-k 라우팅·load balance)과 LGT 블록(local+global 융합) |
| [explanation/saf-fusion.md](explanation/saf-fusion.md) | SAF(Spatial Attention Fusion): cross-attn → spatial → channel → gated fusion 단계별 |
| [explanation/known-issues.md](explanation/known-issues.md) | ⭐ 코드 대조로 확인한 결함·함정 목록(🔴/🟠/🟢) — 학습·인용 전 필독 |

## 관련 자료

- 본 문서의 위키 합성층: obsidian-vault `wiki/domains/mma-former.md`(도메인 MOC).
- 코드 참조는 모두 **심볼명 + 상대 링크**(줄번호 없음)로 한다. 줄번호는 코드 변경에 깨지므로 쓰지 않는다.
