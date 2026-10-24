"""RMUC 2027 decision layer for Arena-1: the sentry (and later the infantry).

The real sentry's stack is  perception -> decision -> navigation / auto-aim.
A behaviour tree normally fills the decision slot; this package trains a policy
for exactly that slot and nothing below it:

    spec       frames, waypoints, action heads (exported with every run)
    inputs     the per-decision snapshot the policy reads, in plain fields that a
               robot fills from the referee system and its own perception
    obs        snapshot -> fixed-size feature vector, action masks, critic state
    executor   macro action -> 10 Hz Arena commands (the nav + auto-aim stand-in)
    bt         hand-written behaviour trees in the same action space (baselines)
    match      one match with any mix of scripts, trees and policies; rewards
"""
