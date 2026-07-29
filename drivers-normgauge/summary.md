Goal: Ship three per-model normalized gauges (aalto-n/comm-n/pool-n; 0=random-pool mean, 1=per-model optimum) on the shipped .standardized frame + one optimizer-usable combined objective with evidence-based weights.
Goal met: yes
Status: done
Blocked by: nothing — branch normgauge is committed+unpushed; pushing/CR is the human gate
Achieved: weights aalto-n 0.5411/comm-n 0.3977/pool-n 0.0612 from HELD-OUT predictive skill (POOL at its measured 0.0612 unique-variance share); gauges+blend scorer in src/keybo/scoring/model_norm.py wired as `keybo optimize --model-weight/--model-anchors`; anchors reproduce (AALTO hits MODELNORM's 10M champion exactly, COMMUNITY/POOL seed spread 0.0); full suite rc=0; 36 tests mutation-controlled; 6 self-kills.
Key finding: "The scheme reorders nothing" was conflating two claims — WITHIN a model normalization reorders nothing (0/66 discordant, rho +1.000000) but ACROSS models the WEIGHT reorders a lot (30/66, rho +0.2448). Also: drop-POOL 50/50 ties the registered weighting, so POOL's weight does no observable work.
