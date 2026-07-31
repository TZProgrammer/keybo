Goal: Ship three per-model normalized gauges (aalto-n/comm-n/pool-n; 0=random-pool mean, 1=per-model optimum) on the shipped .standardized frame + one optimizer-usable combined objective with evidence-based weights.
Goal met: yes
Status: done
Blocked by: nothing — state flushed; branch normgauge @ f2d76f8 (base dd04219) is committed+UNPUSHED in the shared clone's refs, so it survives workspace destruction. Pushing/CR is the human gate.
Achieved: weights aalto-n 0.5411/comm-n 0.3977/pool-n 0.0612 from HELD-OUT predictive skill (POOL at its measured 0.0612 unique-variance share); gauges+blend in src/keybo/scoring/model_norm.py wired as `keybo optimize --model-weight/--model-anchors`; anchors reproduce (AALTO hits MODELNORM's 10M champion exactly); full suite rc=0; 36 tests mutation-controlled; 6 self-kills; recovery recipe RUN (36 passed from a detached checkout).
Key finding: "The scheme reorders nothing" conflated two claims — WITHIN a model normalization reorders nothing (0/66, rho +1.000000); ACROSS models the WEIGHT reorders a lot (30/66, rho +0.2448). And drop-POOL 50/50 TIES the registered weighting, so POOL's weight does no observable work.
