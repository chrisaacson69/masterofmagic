"""
Master of Magic Combat Engine
Ported from the wiki's AdvancedDamageCalculator.js

This is a faithful port of the probability math from the wiki calculator.
The JS source is the authoritative reference — this port should produce
identical results for identical inputs.

Architecture:
  binom_arr()                — binomial probability distribution
  apply_block()              — convolve hits with defense blocks
  get_block_arr()            — block distribution (with invulnerability)
  calc_transition_table()    — physical damage: hits -> blocks -> spill across figures
  construct_gaze_transition_table() — resistance-based gaze attacks
  calc_one_attack()          — apply one attack type to joint HP state
  calcround()                — orchestrate all attack phases in one combat round
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Building blocks — ported from JS
# ---------------------------------------------------------------------------

def binom_arr(to_fail: float, num_rolls: int) -> list[float]:
    """
    Binomial probability distribution.
    Returns P(k successes) for k = 0..num_rolls.
    to_fail = probability of failure (1 - to_hit).

    JS: binom_arr(to_fail, num_rolls)
    """
    n = num_rolls
    arr = [0.0] * (1 + n)
    if to_fail <= 0:
        arr[n] = 1.0
        return arr
    ntmp = to_fail ** n
    arr[0] = ntmp
    for i in range(1, n + 1):
        ntmp *= (n + 1 - i) * (1 - to_fail) / (to_fail * i)
        arr[i] = ntmp
    return arr


def binom_arr_stride(to_fail: float, num_rolls: int, stride: int) -> list[float]:
    """
    Binomial distribution with stride — non-zero entries only at multiples of stride.
    Used for area damage where each success = stride damage.

    JS: binom_arr_stride(to_fail, num_rolls, stride)
    """
    arr = [0.0] * (1 + num_rolls * stride)
    if to_fail <= 0:
        arr[num_rolls * stride] = 1.0
        return arr
    ntmp = to_fail ** num_rolls
    arr[0] = ntmp
    for i in range(1, num_rolls + 1):
        ntmp *= (num_rolls + 1 - i) * (1 - to_fail) / (to_fail * i)
        arr[i * stride] = ntmp
    return arr


def apply_block(att_arr: list[float], block_arr: list[float], offset: int = 0) -> list[float]:
    """
    Convolve attack hit distribution with defense block distribution.
    Net damage = max(hits - blocks, 0).

    offset: skip first `offset` entries of att_arr (used for re-blocking
    after a figure dies and the next figure gets its own defense rolls).

    JS: apply_block(att_arr, block_arr, offset)
    """
    if offset >= len(att_arr):
        return []
    new_len = len(att_arr) - offset
    new_arr = [0.0] * new_len
    for i in range(new_len):
        for j in range(len(block_arr)):
            if j > i:
                new_arr[0] += att_arr[i + offset] * block_arr[j]
            else:
                new_arr[i - j] += att_arr[i + offset] * block_arr[j]
    return new_arr


def get_block_arr(to_fail: float, num_rolls: int, invuln: bool) -> list[float]:
    """
    Defense block distribution, with optional invulnerability.
    Invulnerability adds +2 guaranteed blocks (shifts distribution right by 2).

    JS: get_block_arr(to_fail, num_rolls, invuln)
    """
    if not invuln:
        return binom_arr(to_fail, num_rolls)
    arr = [0.0] * (3 + num_rolls)
    if to_fail <= 0:
        arr[num_rolls + 2] = 1.0
        return arr
    ntmp = to_fail ** num_rolls
    arr[0] = 0.0
    arr[1] = 0.0
    arr[2] = ntmp
    for i in range(1, num_rolls + 1):
        ntmp *= (num_rolls + 1 - i) * (1 - to_fail) / (to_fail * i)
        arr[i + 2] = ntmp
    return arr


# ---------------------------------------------------------------------------
# Transition tables — the core damage model
# ---------------------------------------------------------------------------

def calc_transition_table(
    swords: int,
    att_to_hit: float,
    block_arr: list[float],
    def_fig_max_hp: int,
    def_max_hp: int,
    kill_entire_figs: int = 0,
) -> np.ndarray:
    """
    Build transition table for physical damage.

    Physical damage model:
    1. Attacker rolls `swords` dice at `att_to_hit` chance
    2. Defender blocks with `block_arr` distribution
    3. Net damage spills through figures (each with def_fig_max_hp)
    4. When a figure dies, remaining damage faces fresh defense rolls
       from the next figure (re-blocking via apply_block with offset)

    kill_entire_figs: if 1, damage only kills whole figures (area damage mode).

    Returns flat array indexed as [from_hp * (def_max_hp+1) + to_hp].
    from_hp=0 row is the absorbing "dead" state.

    JS: calc_transition_table(swords, att_to_hit, block_arr, def_fig_max_hp,
                              def_max_hp, transition_table, kill_entire_figs)
    """
    size = def_max_hp + 1
    T = np.zeros(size * size)
    T[0] = 1.0  # dead stays dead

    att_arr_base = binom_arr(1.0 - att_to_hit, swords)

    for hp in range(1, def_max_hp + 1):
        row_offset = hp * size
        att_arr = apply_block(att_arr_base[:], block_arr)
        figs_left = math.ceil((hp - 0.5) / def_fig_max_hp)

        if kill_entire_figs == 0:
            # Physical damage: spills across figures with re-blocking
            idx4 = 0
            partial_sum = 0.0
            for fig in range(figs_left):
                for dmg in range(min(len(att_arr), def_fig_max_hp)):
                    if idx4 >= hp:
                        break
                    partial_sum += att_arr[dmg]
                    T[row_offset + hp - idx4] = att_arr[dmg]
                    idx4 += 1
                if idx4 >= hp:
                    break
                # Re-block: next figure gets fresh defense rolls
                att_arr = apply_block(att_arr, block_arr, def_fig_max_hp)
                if len(att_arr) == 0:
                    break
            if idx4 >= hp:
                T[row_offset] += 1.0 - partial_sum
        else:
            # Area damage: kills whole figures only
            T[row_offset] = 1.0
            for fig_killed in range(min(len(att_arr), figs_left)):
                T[row_offset + hp - fig_killed * def_fig_max_hp] = att_arr[fig_killed]
                T[row_offset] -= att_arr[fig_killed]

    return T


def make_grand_transition_table(att_max_fig_ct: int, def_max_hp_p1: int, T: np.ndarray):
    """
    Expand a single-figure transition table into a grand transition table
    with one sub-table per attacker figure count.

    For multi-figure attackers, each figure count level may have different
    damage output (e.g., immolation scales with attacker figures).
    In the simple case, all sub-tables are identical copies.

    Modifies T in-place by writing additional sub-tables after the first.

    JS: make_grand_transition_table(att_max_fig_ct, def_max_hp_p1, transition_table)
    """
    if att_max_fig_ct <= 1:
        return
    block_size = def_max_hp_p1 * def_max_hp_p1
    # Copy the base table for each figure count
    for fig in range(1, att_max_fig_ct):
        src_start = 0
        dst_start = fig * block_size
        # Ensure T is large enough
        if dst_start + block_size > len(T):
            T_new = np.zeros(dst_start + block_size)
            T_new[:len(T)] = T
            T = T_new
        T[dst_start:dst_start + block_size] = T[src_start:src_start + block_size]


def apply_grand_transition_table_to_side(
    att_side: int,
    att_max_fig_ct: int,
    att_fig_hp: int,
    def_max_hp: int,
    grand_transition_table: np.ndarray,
    joint_hparr: np.ndarray,
) -> None:
    """
    Apply the grand transition table to the joint HP probability array.

    For each attacker HP level (grouped by figure count), applies the
    corresponding sub-table's transitions to the defender's HP distribution.

    att_side: 0 = attacker is side A (rows), 1 = attacker is side B (columns)

    Modifies joint_hparr in-place.

    JS: apply_grand_transition_table_to_side(att_side, att_max_fig_ct,
        att_fig_hp, def_max_hp, grand_transition_table,
        def_tmp_prob_array, joint_hparr)
    """
    att_max_hp_p1 = att_max_fig_ct * att_fig_hp + 1
    def_max_hp_p1 = def_max_hp + 1
    def_tmp = np.zeros(def_max_hp_p1)

    for att_fig_ct in range(1, att_max_fig_ct + 1):
        grand_offset = (att_fig_ct - 1) * def_max_hp_p1 * def_max_hp_p1
        att_hp_lo = 1 + (att_fig_ct - 1) * att_fig_hp
        att_hp_hi = att_fig_ct * att_fig_hp

        for jidx in range(att_hp_lo, att_hp_hi + 1):
            def_tmp[:] = 0.0

            if att_side == 0:
                jrow_offset = jidx * def_max_hp_p1
                for idx in range(def_max_hp_p1):
                    trow_offset = idx * def_max_hp_p1 + grand_offset
                    jconst = joint_hparr[jrow_offset + idx]
                    if jconst == 0:
                        continue
                    for idx2 in range(idx + 1):
                        def_tmp[idx2] += jconst * grand_transition_table[trow_offset + idx2]
                joint_hparr[jrow_offset:jrow_offset + def_max_hp_p1] = def_tmp
            else:
                for idx in range(def_max_hp_p1):
                    trow_offset = idx * def_max_hp_p1 + grand_offset
                    jconst = joint_hparr[idx * att_max_hp_p1 + jidx]
                    if jconst == 0:
                        continue
                    for idx2 in range(idx + 1):
                        def_tmp[idx2] += jconst * grand_transition_table[trow_offset + idx2]
                for idx in range(def_max_hp_p1):
                    joint_hparr[idx * att_max_hp_p1 + jidx] = def_tmp[idx]


def calc_one_attack(
    att_side: int,
    att_max_fig_ct: int,
    att_val: int,
    att_to_hit: float,
    att_fig_hp: int,
    def_max_fig_ct: int,
    shields: int,
    def_toblock: float,
    def_fig_hp: int,
    def_invuln: bool,
    joint_hparr: np.ndarray,
    kill_entire_figs: int = 0,
) -> None:
    """
    Calculate one attack type and apply it to the joint HP state.

    This is the main entry point for physical/area damage.
    Builds the transition table, expands for multi-figure attackers,
    and applies to the joint probability array.

    JS: calc_one_attack(att_side, att_max_fig_ct, att_val, att_to_hit,
        att_fig_hp, def_max_fig_ct, shields, def_toblock, def_fig_hp,
        def_invuln, transition_table, def_tmp_prob_array, joint_hparr,
        kill_entire_figs)
    """
    def_max_hp = def_max_fig_ct * def_fig_hp
    block_arr = get_block_arr(1.0 - def_toblock, shields, def_invuln)

    T = calc_transition_table(
        att_val, att_to_hit, block_arr,
        def_fig_hp, def_max_hp, kill_entire_figs
    )
    make_grand_transition_table(att_max_fig_ct, def_max_hp + 1, T)
    apply_grand_transition_table_to_side(
        att_side, att_max_fig_ct, att_fig_hp,
        def_max_hp, T, joint_hparr
    )


def construct_gaze_transition_table(
    maxfig: int,
    resist: int,
    bless: int,
    chaosnature_def: int,
    max_fighp: int,
    dgaze: int,
    sgaze: int,
    doomgaze: int,
) -> np.ndarray:
    """
    Build transition table for gaze attacks (resistance-based).

    Death Gaze: each figure must resist or die.
    Stoning Gaze: each figure must resist or die (different resistance modifier).
    Doom Gaze: flat HP damage, no rolls.

    JS: construct_gaze_transition_table(maxfig, resist, bless, chaosnature_def,
        max_fighp, dgaze, sgaze, doomgaze, transition_table)
    """
    maxhp = maxfig * max_fighp
    maxhp_p1 = maxhp + 1
    T = np.zeros(maxhp_p1 * maxhp_p1)

    # Initialize as identity (no damage by default)
    for hp in range(maxhp_p1):
        T[hp * maxhp_p1 + hp] = 1.0

    dgaze_resist = resist + 3 * bless
    sgaze_resist = resist + chaosnature_def

    if dgaze > dgaze_resist or sgaze > sgaze_resist:
        survival_prob1 = 1.0
        survival_prob2 = 1.0

        if dgaze - dgaze_resist < 10:
            if dgaze > dgaze_resist:
                survival_prob1 = 0.1 * (10 + dgaze_resist - dgaze)
        else:
            survival_prob1 = 0.0

        if sgaze - sgaze_resist < 10:
            if sgaze > sgaze_resist:
                if survival_prob1 == 1.0:
                    survival_prob1 = 0.1 * (10 + sgaze_resist - sgaze)
                elif survival_prob1 != 0.0:
                    survival_prob2 = 0.1 * (10 + sgaze_resist - sgaze)
        else:
            survival_prob1 = 0.0

        for fig_count in range(1, maxfig + 1):
            if survival_prob2 != 1.0:
                binom_buf2 = binom_arr(1.0 - survival_prob1, fig_count)
                binom_buf = [0.0] * (1 + fig_count)
                ntmp = (1.0 - survival_prob2) ** fig_count
                binom_buf[0] = ntmp
                for i2 in range(fig_count - 1, -1, -1):
                    ntmp *= (1 + i2) * survival_prob2 / ((1.0 - survival_prob2) * (fig_count - i2))
                    for i3 in range(fig_count + 1):
                        if i3 <= i2:
                            binom_buf[0] += ntmp * binom_buf2[i3]
                        else:
                            binom_buf[i3 - i2] += ntmp * binom_buf2[i3]
            else:
                binom_buf = binom_arr(1.0 - survival_prob1, fig_count)

            for hp in range((fig_count - 1) * max_fighp + 1, fig_count * max_fighp + 1):
                row_offset = hp * maxhp_p1
                delta = fig_count * max_fighp - hp
                T[row_offset + 0] = binom_buf[0]
                T[row_offset + hp] = 0.0  # clear identity entry
                for i3 in range(1, fig_count + 1):
                    T[row_offset + i3 * max_fighp - delta] = binom_buf[i3]

    # Doom Gaze: flat damage
    if doomgaze > 0:
        for hp in range(1, maxhp + 1):
            row_offset = hp * maxhp_p1
            for d in range(1, doomgaze + 1):
                T[row_offset + 0] += T[row_offset + d]
                if d == hp:
                    break
            for d in range(1, hp - doomgaze + 1):
                T[row_offset + d] = T[row_offset + d + doomgaze]
            if hp > doomgaze:
                for d in range(hp - doomgaze + 1, hp + 1):
                    T[row_offset + d] = 0.0

    return T


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_against_js_example():
    """
    Quick smoke test: compute a simple matchup and print results.
    Can be compared against the wiki calculator manually.
    """
    # Example: 6 swordsmen (3 melee, 1 HP each, 30% to-hit)
    #   vs 6 swordsmen (2 defense, 30% to-block)
    swords = 3  # per figure, but calc_transition_table takes total
    att_to_hit = 0.30
    shields = 2
    def_to_block = 0.30
    def_fig_hp = 1
    def_figs = 6
    def_max_hp = def_figs * def_fig_hp

    block_arr = get_block_arr(1.0 - def_to_block, shields, False)
    T = calc_transition_table(
        swords * def_figs,  # total swords for one figure attacking
        att_to_hit, block_arr,
        def_fig_hp, def_max_hp
    )

    # Print transition probabilities from full HP
    size = def_max_hp + 1
    row = def_max_hp  # full HP row
    print(f"From {row} HP, damage distribution:")
    for to_hp in range(row, -1, -1):
        prob = T[row * size + to_hp]
        if prob > 0.001:
            dmg = row - to_hp
            print(f"  {dmg} damage: {prob:.4f} ({prob*100:.1f}%)")


if __name__ == "__main__":
    validate_against_js_example()
