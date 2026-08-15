# 실험 러닝 로그

> 계획서 §10: **실패·부정 결과 포함 전부 기록.** 시간 역순이 아니라 정순으로 적는다.

---

## 2026-08-15 — 구현 세션 (GPU 없음)

### 환경
| 항목 | 값 |
|---|---|
| GPU | **없음** (`nvidia-smi` 미설치) |
| CPU / RAM | 4 코어 / 15 GB |
| torch | 2.13.0 (CPU 실행) |
| STEER 핀 | `08add1cc27f4d32a78a6c0d6cb857aa52b8f2a55`, verl `0.4.1.dev` |

**결론: 이 세션에서 학습 실행은 불가능.** 따라서 GPU 없이 검증 가능한 전부
(코드 분석·라이브러리·패치·테스트)를 완료하고, 학습은 미실행으로 남긴다.

### 완료

- `docs/steer_code_map.md` — Phase 0 게이트 산출물. 원본 코드 정독 결과.
- `steer_f/` 6개 모듈, `scripts/` 4개, `patches/` 3개, `run/` 2개.
- 테스트 **213개 전부 통과** (`python -m pytest tests/ -q`).
- 패치 적용 → 되돌리기 → 재적용 왕복 검증 완료.

### 발견 1 — 계획서 §0.1의 STEER 서술이 코드와 다르다 (설계 영향 큼)

계획서는 밴드 `[ΔH_low, ΔH_high]`와 이산 α `{γ, 1, 1/γ}`를 전제하지만,
**실제 구현에는 밴드도 γ도 이산 α도 없다.** `compute_token_weights`는 Ω를
마이크로배치 min-max로 `[token_weight_min, token_weight_max]`에 연속 선형 매핑
(또는 지수 매핑)한다. 상세는 `docs/steer_code_map.md` §3.

영향:
- 계획서 §4.1의 "밴드 분위수 재산정" 절차는 **적용 대상이 없다.** 대응물로
  `token_weight_distribution`(가중치 범위 3등분 점유율)을 로깅한다.
- 계획서 §0.2의 "α의 이산 3단 구조 유지"라는 방어 논리는 성립하지 않는다.
  유계성은 대신 `clip(A_H, ±c)` + `[w_min, w_max]` clamp가 담당한다.

### 발견 2 — 정규화는 z-score가 아니라 RMS여야 한다

계획서 §4.1은 `z_norm`을 쓰라고 하지만, STEER 매핑의 불변성을 조사한 결과
**중심이동(centering)은 λ=0 동치성을 깨뜨린다:**

| 매핑 | 불변인 변환 |
|---|---|
| `linear=True` | 아핀 `a·x+b` (a>0) |
| `linear=False` | 양의 스케일 `a·x`만 (지수는 평행이동 불변 아님) |
| `mode="symmetric"` | `abs()`가 붙어 **0이 의미를 가짐 → 중심이동 불가** |

교집합은 "중심이동 없는 양의 스케일링"뿐. 기본값을 `norm="scale"`(RMS)로 하고,
`z`는 선택 가능하되 `SteerFConfig.validate()`가 위험 조합에서 경고한다.
실패 사례는 `test_z_norm_would_break_symmetric_equivalence`가 실증한다.

### 발견 3 — 원본 verl 포크의 잠재 크래시

`verl/workers/fsdp_workers.py:772`가 3-tuple을 반환하는
`compute_log_prob`에서 **4개를 언패킹**한다. STEER 스크립트는
`use_kl_loss=False`, `use_kl_in_reward=False`라 독립 reference policy가 뜨지
않아 발화하지 않지만, KL 항을 켜는 순간 `ValueError`로 죽는다.
`patches/steerf_forecast_pass.patch`가 반환값을 4개로 늘리면서 부수적으로 해소된다.
KL을 쓸 계획이 없어도, 이 사실을 모르고 켜면 원인 찾기 어려우니 기록해 둔다.

### 발견 4 — 마이크로배치 크기가 하이퍼파라미터다

Ω의 min/max가 **마이크로배치별로** 계산되므로 `ppo_micro_batch_size_per_gpu`를
바꾸면 동일 설정에서도 가중치 분포가 달라진다. Phase 2의 소형 vs 7B 비교에서
반드시 고정하거나 보고서에 명시할 것. `run/run_steerf_small.sh`에 주석으로 고정해 뒀다.

### 발견 5 — 계획서 §3.3의 GT 정의를 그대로 쓰면 γ_H가 부당하게 불리하다

`H_togo^κ`는 **할인된** 합인데 계획서의 `GT_future_entropy`는 할인 없는 합이다.
할인 예보를 무할인 실측에 맞춰 채점하면 `γ_H < 1`이 예보 품질과 무관한 이유로
전부 손해를 본다. `phase1_validate.py`는 (κ, γ_H) 셀마다 **같은 할인**을 적용한
실측치를 주 지표로 쓰고, 계획서 정의는 `gt_sum_h`로 함께 기록한다.

### 미실행 (전부 GPU 필요)

- [ ] **G0**: STEER 소형 재현 (GRPO vs STEER, 100~150 스텝). 엔트로피 곡선 대조.
- [ ] **G1**: 헤드 워밍업 → MC 검증 → (κ, γ_H, 캘리브레이션) 확정.
- [ ] **G2**: λ 스윕 및 7B 확인.
- [ ] **G3**: Llama / EXAONE 전이.
- [ ] Ablation A1–A7.

> 게이트를 하나도 통과하지 않았으므로, 현재 저장소에는 **STEER-F가 효과가 있다는
> 어떤 증거도 없다.** 있는 것은 "λ=0에서 STEER와 정확히 같고, λ>0에서 정의대로
> 동작한다"는 구현 수준의 보증뿐이다.

### Phase 2 미해결 리스크

1. **`_steerf_unembedding()`의 FSDP 경로가 미검증.** FSDP는 forward 밖에서
   파라미터를 샤딩하므로, lm_head 직접 호출이 실패하거나 샤드만 볼 수 있다.
   첫 다중 GPU 실행에서 여기서 죽으면 두 가지 처방:
   (a) 예보를 `FSDP.summon_full_params(..., writeback=False)`로 감싼다,
   (b) unembedding의 비샤딩 사본을 따로 유지한다.
   `patches/steerf_forecast_pass.patch` 헤더와 해당 docstring에 같은 내용을 적어 뒀다.
2. **예보용 추가 forward 1회**의 비용. verl의 기존 패스는 히든을 버리므로
   rmpad 경로를 수술하는 대신 no-grad forward를 한 번 더 도는 쪽을 택했다.
   backward는 없으므로 스텝당 증가는 대략 +1 forward. 실측 후 기록할 것.
3. **MTP 보조 손실(`β_mtp`)이 아직 학습 루프에 연결되지 않았다.** 헤드는 현재
   Phase 1 워밍업 상태로 고정(freeze)되어 돈다. 즉 지금 상태는 사실상
   **Ablation A7(헤드 freeze)** 이며, 계획서 §4.1의 "헤드가 정책과 함께 진화"는
   미구현이다. `mtp_ce_loss`는 준비돼 있고 `update_policy`에서 호출하면 되지만,
   그러려면 학습 forward에서 히든을 꺼내야 해서 리스크 1과 같은 문제를 만난다.
   **G2를 A7 조건에서 먼저 돌리고, 표류(KL) 로그를 보고 필요성을 판단할 것.**
4. `LambdaDriftController`도 같은 이유로 아직 루프에 연결되지 않았다
   (KL 표류 측정에 헤드-정책 동시 분포가 필요). 단위 테스트는 통과 상태.

### 다음 세션 시작 지점

```bash
bash scripts/setup_steer.sh
pip install -e third_party/STEER
python -m pytest tests/ -q                 # 213 passed 확인
ARM=grpo  ./run/run_steerf_small.sh        # G0 절반
ARM=steer ./run/run_steerf_small.sh        # G0 나머지
```

`ARM=steer`가 stock STEER와 비트 단위로 같다는 것은 테스트로 보장되므로,
G0 재현은 원본 레포를 따로 돌릴 필요 없이 이 저장소만으로 끝난다.

---

## (다음 항목은 여기에)
