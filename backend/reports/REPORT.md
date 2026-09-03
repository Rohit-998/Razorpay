# Recovery policy evaluation

400 batch runs · 4 policies × 5 scenarios × 20 seeds · generated 2026-09-03 19:53 UTC

## Measured money recovered

39,200 failed payments worth ₹75.06 cr, across 5 scenarios × 20 seeds. Every policy saw the identical batch at every seed — same customers, same bank outages, same coin flips — so each row below is a paired difference, not two separate experiments.

| Policy | Lift per batch, 95% CI | Share of achievable | Net of spend | Seeds won |
| --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹0 | 0/100 |
| `naive_retry` | ₹6.04 L [₹5.34 L, ₹6.74 L] | 23.6% | ₹6.04 L | 100/100 |
| `rules` | ₹11.70 L [₹10.80 L, ₹12.63 L] | 45.6% | ₹11.68 L | 100/100 |
| `oracle` (ceiling, not a result) | ₹25.64 L [₹24.39 L, ₹26.93 L] | 100.0% | ₹25.63 L | 100/100 |

A batch is one scenario at one seed — roughly 392 failed payments. Lift is recovery minus the same batch under `do_nothing`, so a policy is credited only with money that would not have arrived on its own. Share is against `oracle`, which reads the hidden state and is an upper bound rather than a proposal. An interval that straddles zero would be printed straddling zero.

## Whether it could actually ship

None of these appear in a recovery rate. Each one is a reason a payments team would refuse to deploy a policy however much money it appears to make. Counts are totals over all 100 batches, not per batch.

| Policy | Messages in quiet hours | Instruments we blocked | Actions the gateway refused | Failed to stop | Verdict |
| --- | --- | --- | --- | --- | --- |
| `do_nothing` | 0 | 0 | 0 | 0 | clean |
| `naive_retry` | 8,447 of 29,707 | 270 | 83,833 | 0 | **not shippable** |
| `rules` | 0 | 95 | 0 | 0 | **not shippable** |
| `oracle` | 128 of 20,502 | 11 | 0 | 0 | n/a — cheats by construction |

Quiet hours are 22:00–08:00 IST, judged against when the message was sent rather than when its effects settled. *Instruments we blocked* counts cards and mandates that were working at failure time and were killed by our own retries — customers left worse off than if nothing had been done. The ceiling's own quiet-hour messages are real: nothing forbids them, and for a patient customer at 02:00 the overnight read penalty is occasionally worth paying. It is listed to be honest about what the upper bound is, not held up as a target.

## Scenario by scenario

### `baseline`

₹15.64 cr at risk over 20 seeds, ₹78.22 L per batch. Agent bench: 20 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹24.96 L | ₹0 | 0 | 0 | 0/20 | ₹0 |
| `naive_retry` | ₹6.49 L [₹5.64 L, ₹7.36 L] | 26.0% | ₹18.47 L | ₹64 | 158 | 293 | 0/20 | ₹4.78 L |
| `rules` | ₹12.19 L [₹10.84 L, ₹13.66 L] | 48.8% | ₹12.77 L | ₹1,721 | 69 | 483 | 18/20 | ₹6.57 L |
| `oracle` | ₹24.96 L [₹22.81 L, ₹27.14 L] | 100.0% | ₹0 | ₹968 | 52 | 203 | 10/20 | ₹0 |

### `outage_day`

₹12.93 cr at risk over 20 seeds, ₹64.66 L per batch. Agent bench: 16 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹20.57 L | ₹0 | 0 | 0 | 0/16 | ₹0 |
| `naive_retry` | ₹6.08 L [₹5.10 L, ₹7.13 L] | 29.6% | ₹14.49 L | ₹51 | 126 | 232 | 0/16 | ₹4.31 L |
| `rules` | ₹9.62 L [₹8.30 L, ₹11.09 L] | 46.8% | ₹10.95 L | ₹1,188 | 56 | 389 | 12/16 | ₹5.41 L |
| `oracle` | ₹20.57 L [₹18.36 L, ₹22.83 L] | 100.0% | ₹0 | ₹781 | 42 | 164 | 8/16 | ₹0 |

### `salary_week`

₹12.74 cr at risk over 20 seeds, ₹63.72 L per batch. Agent bench: 19 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹21.52 L | ₹0 | 0 | 0 | 0/19 | ₹0 |
| `naive_retry` | ₹4.19 L [₹3.51 L, ₹4.85 L] | 19.4% | ₹17.34 L | ₹68 | 168 | 308 | 0/19 | ₹2.25 L |
| `rules` | ₹8.78 L [₹7.41 L, ₹10.16 L] | 40.8% | ₹12.74 L | ₹1,672 | 75 | 500 | 17/19 | ₹4.28 L |
| `oracle` | ₹21.52 L [₹20.03 L, ₹23.22 L] | 100.0% | ₹0 | ₹941 | 49 | 197 | 10/19 | ₹0 |

### `festival_spike`

₹21.71 cr at risk over 20 seeds, ₹1.09 cr per batch. Agent bench: 28 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹34.59 L | ₹0 | 0 | 0 | 0/28 | ₹0 |
| `naive_retry` | ₹11.01 L [₹9.70 L, ₹12.33 L] | 31.8% | ₹23.57 L | ₹85 | 200 | 386 | 0/28 | ₹3.87 L |
| `rules` | ₹18.49 L [₹17.16 L, ₹20.04 L] | 53.5% | ₹16.10 L | ₹2,279 | 91 | 650 | 23/28 | ₹9.20 L |
| `oracle` | ₹34.59 L [₹33.11 L, ₹36.11 L] | 100.0% | ₹0 | ₹1,294 | 71 | 277 | 13/28 | ₹0 |

### `stress_dead_instruments`

₹12.04 cr at risk over 20 seeds, ₹60.18 L per batch. Agent bench: 15 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹26.58 L | ₹0 | 0 | 0 | 0/15 | ₹0 |
| `naive_retry` | ₹2.43 L [₹1.96 L, ₹2.89 L] | 9.1% | ₹24.15 L | ₹58 | 170 | 265 | 0/15 | ₹3.03 L |
| `rules` | ₹9.42 L [₹8.51 L, ₹10.38 L] | 35.4% | ₹17.16 L | ₹1,427 | 43 | 362 | 15/15 | ₹3.96 L |
| `oracle` | ₹26.58 L [₹25.12 L, ₹28.05 L] | 100.0% | ₹0 | ₹695 | 49 | 185 | 7/15 | ₹0 |

*Unprovable* is money that arrived through the customer's own channel within six hours of us messaging them. It is never counted as a win — a policy that messages everyone accumulates a large pile of it and has recovered nothing.

## By root cause — `rules` against the ceiling

Pooled over every scenario and seed. The root cause is latent: it is what the environment used to decide which physical precondition was false, and no policy is shown it.

| Root cause | Payments | At risk | `rules` recovered | Ceiling recovered | Gap |
| --- | --- | --- | --- | --- | --- |
| `BANK_DOWNTIME` | 6,753 | ₹16.96 cr | ₹10.70 cr (63%) | ₹13.39 cr (79%) | ₹2.69 cr |
| `AUTH_TIMEOUT` | 7,465 | ₹15.98 cr | ₹11.87 cr (74%) | ₹12.68 cr (79%) | ₹80.60 L |
| `PERMANENT_DECLINE` | 5,617 | ₹12.03 cr | ₹3.49 cr (29%) | ₹8.13 cr (68%) | ₹4.63 cr |
| `NETWORK_TRANSIENT` | 5,145 | ₹9.93 cr | ₹8.57 cr (86%) | ₹8.98 cr (90%) | ₹40.73 L |
| `INSUFFICIENT_FUNDS` | 8,584 | ₹7.90 cr | ₹2.14 cr (27%) | ₹3.44 cr (44%) | ₹1.30 cr |
| `MERCHANT_ERROR` | 2,265 | ₹7.36 cr | ₹1.79 cr (24%) | ₹4.84 cr (66%) | ₹3.06 cr |
| `WRONG_CREDENTIALS` | 3,371 | ₹4.90 cr | ₹2.48 cr (51%) | ₹3.53 cr (72%) | ₹1.05 cr |

A negative gap is not an error and not `rules` beating the ceiling. The ceiling maximises the batch, not each cause: with a finite agent bench and a deadline it will spend an hour on a large `PERMANENT_DECLINE` instead of a `NETWORK_TRANSIENT` that `rules` picks up by reflex. The ceiling is only guaranteed to be an upper bound on the total, which is where it is used.

## How these numbers were produced

**The headline is measured twice, by routes that share no arithmetic.** Once as recovery minus the same batch under `do_nothing`. Once from the environment's own private verdict on causation, which it assigns per payment without knowing what any policy did. The second can only ever be the larger of the two, and the gap is not error: it is money the policy collected on Monday that was going to arrive on Wednesday anyway. Real, faster, and deliberately kept out of the headline.

| Policy | Recovered sooner than it would have arrived, per batch |
| --- | --- |
| `do_nothing` | ₹0 |
| `naive_retry` | ₹3.67 L |
| `rules` | ₹3.96 L |
| `oracle` | ₹0 |

**Consistency check across every run: passed.** No policy's lift exceeded the recovery the environment attributes to it.

**The intervals are bootstrap, not t-based.** Rupee lift is a sum over a heavy-tailed amount distribution, and one payment can be several percent of a batch's whole figure, so the interval is resampled from the paired per-seed differences rather than assumed normal. The resampling seed is fixed, so regenerating this report reproduces the same bounds.

**Reproducing it:**

```bash
cd backend && python -m app.eval
```
