# Recovery policy evaluation

500 batch runs · 5 policies × 5 scenarios × 20 seeds · generated 2026-09-05 13:34 UTC

## Measured money recovered

39,200 failed payments worth ₹75.06 cr, across 5 scenarios × 20 seeds. Every policy saw the identical batch at every seed — same customers, same bank outages, same coin flips — so each row below is a paired difference, not two separate experiments.

| Policy | Lift per batch, 95% CI | Share of achievable | Net of spend | Seeds won |
| --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹0 | 0/100 |
| `naive_retry` | ₹6.04 L [₹5.34 L, ₹6.74 L] | 23.6% | ₹6.04 L | 100/100 |
| `rules` | ₹11.70 L [₹10.80 L, ₹12.63 L] | 45.6% | ₹11.68 L | 100/100 |
| `payrevive` | ₹17.52 L [₹16.45 L, ₹18.61 L] | 68.3% | ₹17.51 L | 100/100 |
| `oracle` (ceiling, not a result) | ₹25.64 L [₹24.39 L, ₹26.93 L] | 100.0% | ₹25.63 L | 100/100 |

A batch is one scenario at one seed — roughly 392 failed payments. Lift is recovery minus the same batch under `do_nothing`, so a policy is credited only with money that would not have arrived on its own. Share is against `oracle`, which reads the hidden state and is an upper bound rather than a proposal. An interval that straddles zero would be printed straddling zero.

## Whether it could actually ship

None of these appear in a recovery rate. The first four are gates — zero is attainable for each, so any count above zero is a defect and no amount of lift buys it back. The fifth is a cost: a failed retry can kill a working card whatever the reason for the failure, so the only policy that blocks nothing is one that retries nothing. Counts are totals over all 100 batches.

| Policy | Messages in quiet hours | Actions the gateway refused | Actions compliance would refuse | Failed to stop | Working instruments we blocked | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | 0 | 0 | 0 | 0 | 0 | clean |
| `naive_retry` | 8,447 of 29,707 | 83,833 | 8,800 of 46,174 | 0 | 270 of 33,583 (0.80%) | **fails a gate** |
| `rules` | 0 | 0 | 2,445 of 56,053 | 0 | 95 of 33,583 (0.28%) | **fails a gate** |
| `payrevive` | 0 | 0 | 0 | 0 | 72 of 33,583 (0.21%) | shippable · 0.21% harm rate |
| `oracle` | 128 of 20,502 | 0 | 753 of 26,738 | 0 | 11 of 33,583 (0.03%) | n/a — cheats by construction |

Quiet hours are 22:00–08:00 IST, judged against when the message was sent rather than when its effects settled. *Actions compliance would refuse* is the strictest column here and the only one that can invalidate the money in the table above it: every action each policy took is replayed through `app/execution/compliance.py` — the same pure function the live API calls on real traffic, not a re-implementation of it — and counted if it would be blocked at the door. Lift earned by an action production refuses to take is lift that cannot be banked, so a policy with a count here is quoting a number it could not collect. *Working instruments we blocked* counts cards and mandates that were alive at failure time and were killed by our own retries — customers left worse off than if nothing had been done — as a share of the instruments that were alive to be broken. It is the one number here that is underwritten rather than forbidden, and the comparison that matters is against the incumbent: a policy is worth deploying if it breaks fewer instruments *and* recovers more money, which is a stronger claim than either half alone. The ceiling's own quiet-hour messages are real: nothing forbids them, and for a patient customer at 02:00 the overnight read penalty is occasionally worth paying. It is listed to be honest about what the upper bound is, not held up as a target.

## Scenario by scenario

### `baseline`

₹15.64 cr at risk over 20 seeds, ₹78.22 L per batch. Agent bench: 20 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹24.96 L | ₹0 | 0 | 0 | 0/20 | ₹0 |
| `naive_retry` | ₹6.49 L [₹5.64 L, ₹7.36 L] | 26.0% | ₹18.47 L | ₹64 | 158 | 293 | 0/20 | ₹4.78 L |
| `rules` | ₹12.19 L [₹10.84 L, ₹13.66 L] | 48.8% | ₹12.77 L | ₹1,721 | 69 | 483 | 18/20 | ₹6.57 L |
| `payrevive` | ₹17.05 L [₹15.49 L, ₹18.83 L] | 68.3% | ₹7.91 L | ₹1,490 | 89 | 357 | 15/20 | ₹6.97 L |
| `oracle` | ₹24.96 L [₹22.81 L, ₹27.14 L] | 100.0% | ₹0 | ₹968 | 52 | 203 | 10/20 | ₹0 |

### `outage_day`

₹12.93 cr at risk over 20 seeds, ₹64.66 L per batch. Agent bench: 16 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹20.57 L | ₹0 | 0 | 0 | 0/16 | ₹0 |
| `naive_retry` | ₹6.08 L [₹5.10 L, ₹7.13 L] | 29.6% | ₹14.49 L | ₹51 | 126 | 232 | 0/16 | ₹4.31 L |
| `rules` | ₹9.62 L [₹8.30 L, ₹11.09 L] | 46.8% | ₹10.95 L | ₹1,188 | 56 | 389 | 12/16 | ₹5.41 L |
| `payrevive` | ₹13.61 L [₹11.83 L, ₹15.45 L] | 66.2% | ₹6.96 L | ₹1,162 | 72 | 286 | 12/16 | ₹6.52 L |
| `oracle` | ₹20.57 L [₹18.36 L, ₹22.83 L] | 100.0% | ₹0 | ₹781 | 42 | 164 | 8/16 | ₹0 |

### `salary_week`

₹12.74 cr at risk over 20 seeds, ₹63.72 L per batch. Agent bench: 19 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹21.52 L | ₹0 | 0 | 0 | 0/19 | ₹0 |
| `naive_retry` | ₹4.19 L [₹3.51 L, ₹4.85 L] | 19.4% | ₹17.34 L | ₹68 | 168 | 308 | 0/19 | ₹2.25 L |
| `rules` | ₹8.78 L [₹7.41 L, ₹10.16 L] | 40.8% | ₹12.74 L | ₹1,672 | 75 | 500 | 17/19 | ₹4.28 L |
| `payrevive` | ₹13.69 L [₹12.34 L, ₹15.03 L] | 63.6% | ₹7.84 L | ₹1,462 | 94 | 341 | 15/19 | ₹4.71 L |
| `oracle` | ₹21.52 L [₹20.03 L, ₹23.22 L] | 100.0% | ₹0 | ₹941 | 49 | 197 | 10/19 | ₹0 |

### `festival_spike`

₹21.71 cr at risk over 20 seeds, ₹1.09 cr per batch. Agent bench: 28 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹34.59 L | ₹0 | 0 | 0 | 0/28 | ₹0 |
| `naive_retry` | ₹11.01 L [₹9.70 L, ₹12.33 L] | 31.8% | ₹23.57 L | ₹85 | 200 | 386 | 0/28 | ₹3.87 L |
| `rules` | ₹18.49 L [₹17.16 L, ₹20.04 L] | 53.5% | ₹16.10 L | ₹2,279 | 91 | 650 | 23/28 | ₹9.20 L |
| `payrevive` | ₹25.09 L [₹23.53 L, ₹26.76 L] | 72.5% | ₹9.50 L | ₹2,015 | 119 | 494 | 20/28 | ₹11.89 L |
| `oracle` | ₹34.59 L [₹33.11 L, ₹36.11 L] | 100.0% | ₹0 | ₹1,294 | 71 | 277 | 13/28 | ₹0 |

### `stress_dead_instruments`

₹12.04 cr at risk over 20 seeds, ₹60.18 L per batch. Agent bench: 15 calls per batch. Every column below is per batch.

| Policy | Lift, 95% CI | Share | Regret | Spend | Retries | Messages | Agent calls | Unprovable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `do_nothing` | ₹0 [₹0, ₹0] | 0.0% | ₹26.58 L | ₹0 | 0 | 0 | 0/15 | ₹0 |
| `naive_retry` | ₹2.43 L [₹1.96 L, ₹2.89 L] | 9.1% | ₹24.15 L | ₹58 | 170 | 265 | 0/15 | ₹3.03 L |
| `rules` | ₹9.42 L [₹8.51 L, ₹10.38 L] | 35.4% | ₹17.16 L | ₹1,427 | 43 | 362 | 15/15 | ₹3.96 L |
| `payrevive` | ₹18.18 L [₹17.03 L, ₹19.42 L] | 68.4% | ₹8.40 L | ₹1,340 | 84 | 294 | 14/15 | ₹3.64 L |
| `oracle` | ₹26.58 L [₹25.12 L, ₹28.05 L] | 100.0% | ₹0 | ₹695 | 49 | 185 | 7/15 | ₹0 |

*Unprovable* is money that arrived through the customer's own channel within six hours of us messaging them. It is never counted as a win — a policy that messages everyone accumulates a large pile of it and has recovered nothing.

## By root cause — `payrevive` against the ceiling

Pooled over every scenario and seed. The root cause is latent: it is what the environment used to decide which physical precondition was false, and no policy is shown it.

| Root cause | Payments | At risk | `rules` | `payrevive` | Ceiling | Gap left |
| --- | --- | --- | --- | --- | --- | --- |
| `BANK_DOWNTIME` | 6,753 | ₹16.96 cr | ₹10.70 cr (63%) | **₹11.50 cr (68%)** | ₹13.39 cr (79%) | ₹1.89 cr |
| `AUTH_TIMEOUT` | 7,465 | ₹15.98 cr | ₹11.87 cr (74%) | **₹12.22 cr (76%)** | ₹12.68 cr (79%) | ₹45.03 L |
| `PERMANENT_DECLINE` | 5,617 | ₹12.03 cr | ₹3.49 cr (29%) | **₹5.99 cr (50%)** | ₹8.13 cr (68%) | ₹2.14 cr |
| `NETWORK_TRANSIENT` | 5,145 | ₹9.93 cr | ₹8.57 cr (86%) | **₹8.64 cr (87%)** | ₹8.98 cr (90%) | ₹34.43 L |
| `INSUFFICIENT_FUNDS` | 8,584 | ₹7.90 cr | ₹2.14 cr (27%) | **₹2.70 cr (34%)** | ₹3.44 cr (44%) | ₹74.47 L |
| `MERCHANT_ERROR` | 2,265 | ₹7.36 cr | ₹1.79 cr (24%) | **₹3.01 cr (41%)** | ₹4.84 cr (66%) | ₹1.84 cr |
| `WRONG_CREDENTIALS` | 3,371 | ₹4.90 cr | ₹2.48 cr (51%) | **₹2.81 cr (57%)** | ₹3.53 cr (72%) | ₹71.75 L |

A negative gap is not an error and not `payrevive` beating the ceiling. The ceiling maximises the batch, not each cause: with a finite agent bench and a deadline it will spend an hour on a large `PERMANENT_DECLINE` instead of a `NETWORK_TRANSIENT` that `payrevive` picks up by reflex. The ceiling is only guaranteed to be an upper bound on the total, which is where it is used.

**Read across the two policy columns, not down them.** Of the money `rules` left on the table, `payrevive` recovered most where the failure needed a diagnosis — `PERMANENT_DECLINE` (54%), `AUTH_TIMEOUT` (44%), `INSUFFICIENT_FUNDS` (43%) — and least where `rules` was already close to the ceiling and there was nothing left to learn: `NETWORK_TRANSIENT` (15%), `BANK_DOWNTIME` (30%). That is the shape a policy produces when it is reading the failure; a policy that had merely raised its action count would show a flat share across every cause, and one that had found a hole in the simulator would show its largest gains on the causes with the least headroom.

## How these numbers were produced

**The headline is measured twice, by routes that share no arithmetic.** Once as recovery minus the same batch under `do_nothing`. Once from the environment's own private verdict on causation, which it assigns per payment without knowing what any policy did. The second can only ever be the larger of the two, and the gap is not error: it is money the policy collected on Monday that was going to arrive on Wednesday anyway. Real, faster, and deliberately kept out of the headline.

| Policy | Recovered sooner than it would have arrived, per batch |
| --- | --- |
| `do_nothing` | ₹0 |
| `naive_retry` | ₹3.67 L |
| `rules` | ₹3.96 L |
| `payrevive` | ₹5.34 L |
| `oracle` | ₹0 |

**Consistency check across every run: passed.** No policy's lift exceeded the recovery the environment attributes to it.

**The intervals are bootstrap, not t-based.** Rupee lift is a sum over a heavy-tailed amount distribution, and one payment can be several percent of a batch's whole figure, so the interval is resampled from the paired per-seed differences rather than assumed normal. The resampling seed is fixed, so regenerating this report reproduces the same bounds.

**Reproducing it:**

```bash
cd backend && python -m app.eval
```
