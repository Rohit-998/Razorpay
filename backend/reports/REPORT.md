# Recovery policy evaluation

500 batch runs · 5 policies × 5 scenarios × 20 seeds · generated 2026-09-03 22:54 UTC

## Measured money recovered

39,200 failed payments worth ₹75.06 cr, across 5 scenarios × 20 seeds. Every policy saw the identical batch at every seed — same customers, same bank outages, same coin flips — so each row below is a paired difference, not two separate experiments.

| Policy | Lift per batch, 95% CI | Share of achievable | Net of spend | Seeds won |
| --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹0 | 0/100 |
| `naive_retry` | ₹6.04 L [₹5.34 L, ₹6.74 L] | 23.6% | ₹6.04 L | 100/100 |
| `rules` | ₹11.70 L [₹10.80 L, ₹12.63 L] | 45.6% | ₹11.68 L | 100/100 |
| `payrevive` | ₹18.33 L [₹17.23 L, ₹19.44 L] | 71.5% | ₹18.32 L | 100/100 |
| `oracle` (ceiling, not a result) | ₹25.64 L [₹24.39 L, ₹26.93 L] | 100.0% | ₹25.63 L | 100/100 |

A batch is one scenario at one seed — roughly 392 failed payments. Lift is recovery minus the same batch under `do_nothing`, so a policy is credited only with money that would not have arrived on its own. Share is against `oracle`, which reads the hidden state and is an upper bound rather than a proposal. An interval that straddles zero would be printed straddling zero.

## Whether it could actually ship

None of these appear in a recovery rate. The first three are gates — zero is attainable for each, so any count above zero is a defect and no amount of lift buys it back. The fourth is a cost: a failed retry can kill a working card whatever the reason for the failure, so the only policy that blocks nothing is one that retries nothing. Counts are totals over all 100 batches.

| Policy | Messages in quiet hours | Actions the gateway refused | Failed to stop | Working instruments we blocked | Verdict |
| --- | --- | --- | --- | --- | --- |
| `do_nothing` | 0 | 0 | 0 | 0 | clean |
| `naive_retry` | 8,447 of 29,707 | 83,833 | 0 | 270 of 33,583 (0.80%) | **fails a gate** |
| `rules` | 0 | 0 | 0 | 95 of 33,583 (0.28%) | shippable · 0.28% harm rate |
| `payrevive` | 0 | 0 | 0 | 80 of 33,583 (0.24%) | shippable · 0.24% harm rate |
| `oracle` | 128 of 20,502 | 0 | 0 | 11 of 33,583 (0.03%) | n/a — cheats by construction |

Quiet hours are 22:00–08:00 IST, judged against when the message was sent rather than when its effects settled. *Working instruments we blocked* counts cards and mandates that were alive at failure time and were killed by our own retries — customers left worse off than if nothing had been done — as a share of the instruments that were alive to be broken. It is the one number here that is underwritten rather than forbidden, and the comparison that matters is against the incumbent: a policy is worth deploying if it breaks fewer instruments *and* recovers more money, which is a stronger claim than either half alone. The ceiling's own quiet-hour messages are real: nothing forbids them, and for a patient customer at 02:00 the overnight read penalty is occasionally worth paying. It is listed to be honest about what the upper bound is, not held up as a target.

## Scenario by scenario

### `baseline`

₹15.64 cr at risk over 20 seeds, ₹78.22 L per batch. Agent bench: 20 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹24.96 L | ₹0 | 0 | 0 | 0/20 | ₹0 |
| `naive_retry` | ₹6.49 L [₹5.64 L, ₹7.36 L] | 26.0% | ₹18.47 L | ₹64 | 158 | 293 | 0/20 | ₹4.78 L |
| `rules` | ₹12.19 L [₹10.84 L, ₹13.66 L] | 48.8% | ₹12.77 L | ₹1,721 | 69 | 483 | 18/20 | ₹6.57 L |
| `payrevive` | ₹18.05 L [₹16.03 L, ₹20.31 L] | 72.3% | ₹6.91 L | ₹1,246 | 92 | 377 | 12/20 | ₹5.01 L |
| `oracle` | ₹24.96 L [₹22.81 L, ₹27.14 L] | 100.0% | ₹0 | ₹968 | 52 | 203 | 10/20 | ₹0 |

### `outage_day`

₹12.93 cr at risk over 20 seeds, ₹64.66 L per batch. Agent bench: 16 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹20.57 L | ₹0 | 0 | 0 | 0/16 | ₹0 |
| `naive_retry` | ₹6.08 L [₹5.10 L, ₹7.13 L] | 29.6% | ₹14.49 L | ₹51 | 126 | 232 | 0/16 | ₹4.31 L |
| `rules` | ₹9.62 L [₹8.30 L, ₹11.09 L] | 46.8% | ₹10.95 L | ₹1,188 | 56 | 389 | 12/16 | ₹5.41 L |
| `payrevive` | ₹14.09 L [₹12.44 L, ₹15.70 L] | 68.5% | ₹6.48 L | ₹934 | 75 | 302 | 9/16 | ₹5.44 L |
| `oracle` | ₹20.57 L [₹18.36 L, ₹22.83 L] | 100.0% | ₹0 | ₹781 | 42 | 164 | 8/16 | ₹0 |

### `salary_week`

₹12.74 cr at risk over 20 seeds, ₹63.72 L per batch. Agent bench: 19 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹21.52 L | ₹0 | 0 | 0 | 0/19 | ₹0 |
| `naive_retry` | ₹4.19 L [₹3.51 L, ₹4.85 L] | 19.4% | ₹17.34 L | ₹68 | 168 | 308 | 0/19 | ₹2.25 L |
| `rules` | ₹8.78 L [₹7.41 L, ₹10.16 L] | 40.8% | ₹12.74 L | ₹1,672 | 75 | 500 | 17/19 | ₹4.28 L |
| `payrevive` | ₹14.46 L [₹13.12 L, ₹15.90 L] | 67.2% | ₹7.06 L | ₹1,271 | 97 | 362 | 13/19 | ₹3.64 L |
| `oracle` | ₹21.52 L [₹20.03 L, ₹23.22 L] | 100.0% | ₹0 | ₹941 | 49 | 197 | 10/19 | ₹0 |

### `festival_spike`

₹21.71 cr at risk over 20 seeds, ₹1.09 cr per batch. Agent bench: 28 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹34.59 L | ₹0 | 0 | 0 | 0/28 | ₹0 |
| `naive_retry` | ₹11.01 L [₹9.70 L, ₹12.33 L] | 31.8% | ₹23.57 L | ₹85 | 200 | 386 | 0/28 | ₹3.87 L |
| `rules` | ₹18.49 L [₹17.16 L, ₹20.04 L] | 53.5% | ₹16.10 L | ₹2,279 | 91 | 650 | 23/28 | ₹9.20 L |
| `payrevive` | ₹25.46 L [₹23.92 L, ₹27.19 L] | 73.6% | ₹9.13 L | ₹1,619 | 123 | 519 | 16/28 | ₹9.23 L |
| `oracle` | ₹34.59 L [₹33.11 L, ₹36.11 L] | 100.0% | ₹0 | ₹1,294 | 71 | 277 | 13/28 | ₹0 |

### `stress_dead_instruments`

₹12.04 cr at risk over 20 seeds, ₹60.18 L per batch. Agent bench: 15 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹26.58 L | ₹0 | 0 | 0 | 0/15 | ₹0 |
| `naive_retry` | ₹2.43 L [₹1.96 L, ₹2.89 L] | 9.1% | ₹24.15 L | ₹58 | 170 | 265 | 0/15 | ₹3.03 L |
| `rules` | ₹9.42 L [₹8.51 L, ₹10.38 L] | 35.4% | ₹17.16 L | ₹1,427 | 43 | 362 | 15/15 | ₹3.96 L |
| `payrevive` | ₹19.59 L [₹18.16 L, ₹21.06 L] | 73.7% | ₹6.99 L | ₹1,300 | 86 | 317 | 13/15 | ₹3.36 L |
| `oracle` | ₹26.58 L [₹25.12 L, ₹28.05 L] | 100.0% | ₹0 | ₹695 | 49 | 185 | 7/15 | ₹0 |

*Unprovable* is money that arrived through the customer's own channel within six hours of us messaging them. It is never counted as a win — a policy that messages everyone accumulates a large pile of it and has recovered nothing.

## By root cause — `payrevive` against the ceiling

Pooled over every scenario and seed. The root cause is latent: it is what the environment used to decide which physical precondition was false, and no policy is shown it.

| Root cause | Payments | At risk | `rules` | `payrevive` | Ceiling | Gap left |
| --- | --- | --- | --- | --- | --- | --- |
| `BANK_DOWNTIME` | 6,753 | ₹16.96 cr | ₹10.70 cr (63%) | **₹11.63 cr (69%)** | ₹13.39 cr (79%) | ₹1.75 cr |
| `AUTH_TIMEOUT` | 7,465 | ₹15.98 cr | ₹11.87 cr (74%) | **₹12.11 cr (76%)** | ₹12.68 cr (79%) | ₹56.80 L |
| `PERMANENT_DECLINE` | 5,617 | ₹12.03 cr | ₹3.49 cr (29%) | **₹6.21 cr (52%)** | ₹8.13 cr (68%) | ₹1.91 cr |
| `NETWORK_TRANSIENT` | 5,145 | ₹9.93 cr | ₹8.57 cr (86%) | **₹8.62 cr (87%)** | ₹8.98 cr (90%) | ₹35.63 L |
| `INSUFFICIENT_FUNDS` | 8,584 | ₹7.90 cr | ₹2.14 cr (27%) | **₹2.73 cr (35%)** | ₹3.44 cr (44%) | ₹71.50 L |
| `MERCHANT_ERROR` | 2,265 | ₹7.36 cr | ₹1.79 cr (24%) | **₹3.43 cr (47%)** | ₹4.84 cr (66%) | ₹1.42 cr |
| `WRONG_CREDENTIALS` | 3,371 | ₹4.90 cr | ₹2.48 cr (51%) | **₹2.94 cr (60%)** | ₹3.53 cr (72%) | ₹58.78 L |

A negative gap is not an error and not `payrevive` beating the ceiling. The ceiling maximises the batch, not each cause: with a finite agent bench and a deadline it will spend an hour on a large `PERMANENT_DECLINE` instead of a `NETWORK_TRANSIENT` that `payrevive` picks up by reflex. The ceiling is only guaranteed to be an upper bound on the total, which is where it is used.

**Read across the two policy columns, not down them.** Of the money `rules` left on the table, `payrevive` recovered most where the failure needed a diagnosis — `PERMANENT_DECLINE` (59%), `MERCHANT_ERROR` (54%), `INSUFFICIENT_FUNDS` (45%) — and least where `rules` was already close to the ceiling and there was nothing left to learn: `NETWORK_TRANSIENT` (13%), `AUTH_TIMEOUT` (30%). That is the shape a policy produces when it is reading the failure; a policy that had merely raised its action count would show a flat share across every cause, and one that had found a hole in the simulator would show its largest gains on the causes with the least headroom.

## How these numbers were produced

**The headline is measured twice, by routes that share no arithmetic.** Once as recovery minus the same batch under `do_nothing`. Once from the environment's own private verdict on causation, which it assigns per payment without knowing what any policy did. The second can only ever be the larger of the two, and the gap is not error: it is money the policy collected on Monday that was going to arrive on Wednesday anyway. Real, faster, and deliberately kept out of the headline.

| Policy | Recovered sooner than it would have arrived, per batch |
| --- | --- |
| `do_nothing` | ₹0 |
| `naive_retry` | ₹3.67 L |
| `rules` | ₹3.96 L |
| `payrevive` | ₹3.60 L |
| `oracle` | ₹0 |

**Consistency check across every run: passed.** No policy's lift exceeded the recovery the environment attributes to it.

**The intervals are bootstrap, not t-based.** Rupee lift is a sum over a heavy-tailed amount distribution, and one payment can be several percent of a batch's whole figure, so the interval is resampled from the paired per-seed differences rather than assumed normal. The resampling seed is fixed, so regenerating this report reproduces the same bounds.

**Reproducing it:**

```bash
cd backend && python -m app.eval
```
