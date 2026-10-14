# Walkthrough: understand and own this prototype

Work through this in order (about 3–4 hours). By the end you should be able to explain every result in the README and change the code without help.

## Step 1: the math in 15 minutes
1. **Zonotope:** `{c + Gξ : ξ ∈ [-1,1]^m}`. The center `c` plus a weighted sum of "uncertainty directions" (columns of G). Each column is **one source of uncertainty**, e.g. DB query time.
2. **Constrained zonotope:** add `Aξ = b`. This says which *combinations* are possible, e.g. "token check and session lookup can't both be at their worst".
3. **Support function:** how far the set reaches in direction `w` is `wᵀc + max_ξ (wᵀG)ξ`. Without constraints the max is `Σ|wᵀg_k|`: just add up the absolute contributions. With constraints it's a small LP.
4. **Repair:** multiply generator k by `σ_k` (shrink or widen), shift centers, widen targets, relax bounds. Each is a **lever** owned by a team, with a price.
5. **The trick (`problem.py`, `support_expr`).** The max in (3) has a dual LP. In the dual, `σ` appears only on the right-hand side (`Aᵀλ + μ⁺ − μ⁻ = a∘σ`), so "the requirement holds after repair" stays **linear** in σ. One LP then finds the cheapest repair *and* the proof (λ, μ). **Weak duality** makes any dual-feasible point an upper bound, so the requirement provably holds. **Strong duality** means the bound is exact, so we don't over-repair. `test_constrained_repair_is_certified_and_tight` checks exactly this.

➡️ **Exercise:** on paper, compute the checkout spread in E1 without the Auth constraint (`2·1.5 + 1.0 + 0.7 + 1.0 + 2·0.6`). Then with the constraint, Auth's contribution drops from 1.7 to 0.65 (`ξ_t = 1, ξ_s = −0.5`). Why?

## Step 2: read the code (in this order)
| File | What to look for |
|---|---|
| `costrepair/lp.py` | How a constraint is stored as `{var: coeff}`; how the dual objective is rebuilt from HiGHS marginals. |
| `costrepair/architecture.py` | `T = (I − L)⁻¹`: why uncertainty from Infra shows up in Checkout. Run `test_mapping_propagation_matches_manual`. |
| `costrepair/problem.py` | `lever(...)` (a frozen team means upper bound 0) · `support_expr` (unconstrained part = Σ\|a_k\|σ_k; constrained part = dual) · the four requirement types · the cost models (PWL = max of affine pieces). |
| `costrepair/verify.py` | Why we re-check with *primal* LPs: an independent second opinion. |
| `costrepair/queries.py` | Price of restriction = restricted optimum − free optimum. |

## Step 3: run and change things
```bash
cd ~/Desktop/Thesis/prototype_cost_aware_repair
python -m pytest -q tests
python experiments/e1_webshop.py
```
Then try these, predicting the answer **before** you run:
1. In `e1_webshop.py`, make Auth's `shrink` cost 5,000 instead of 500. Who fixes it now?
2. Set Product's `relax` price to 100. What happens to the requirement, and why is that a dangerous answer to give a manager?
3. Change the PWL breakpoint from 500 to 2,000. Why does the solution go back to the unit-cost one?
4. In `e2_iris.py`, set `PER_SCENARIO = 20` and only `"containment"`. Why is "main team differs" higher for containment than for overlap? (Hint: containment needs *every* realization inside, so many more levers become relevant.)

## Step 4: the numbers you must know by heart
- E1: €13,050 (smallest) vs. €2,050 (cheapest), 6.4×; Auth frozen → +€2,700.
- E2: 2,654 configurations from their scenarios and samples; main team differs 52% (overlap) / 74% (containment); median overpay 1.5× / 2.2×; 18% single point of failure under containment.
- E3: the two drivers are the price–contribution correlation (OR 1.83) and the balance of contributions (OR 1.70). Topology doesn't matter by itself.

## Step 5: questions a supervisor will ask (practise answering out loud)
1. *"Why an LP and not their surrogate or gradient search?"* Their recourse targets the probabilistic I(θ) and is approximate. With costs that are linear or convex piecewise-linear, the certified notions (containment, overlap) stay exact LPs with optimality proofs, which matches "keeping certified results".
2. *"What does your containment have to do with their I(θ) ≤ 0.5?"* Containment = I = 0 (stronger), overlap = I < 1 (weaker). They bracket the threshold. A chance-constrained version is future work.
3. *"Is the dual reformulation exact?"* Yes, by strong LP duality when the nominal set is non-empty. The tightness test checks it numerically.
4. *"Where would this break?"* Containment in a general *zonotope* target is coNP-hard (Froese et al.), which is why targets are boxes or polytopes. Scaling generators that appear in `Aξ = b` makes the overlap witness bilinear, which is why I freeze them there. And the prices are synthetic.
5. *"What would you do first in the thesis?"* Get the 11-domain corpus and the manuscript, port the LP into the MATLAB/CORA tool, agree on realistic cost archetypes with you, then rerun E2/E3 on the real corpus.
