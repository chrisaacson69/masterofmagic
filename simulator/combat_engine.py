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
from dataclasses import dataclass, field


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


def make_grand_transition_table(att_max_fig_ct: int, def_max_hp_p1: int, T: np.ndarray) -> np.ndarray:
    """
    Convert a plain transition table into a sequence of them,
    one for each possible number of attacking figures.

    M^n = M^{n-1} * M^1  (matrix power of the base transition table).

    Returns T (possibly reallocated if it needed to grow).

    JS: make_grand_transition_table(att_max_fig_ct, def_max_hp_p1, grand_transition_table)
    """
    if att_max_fig_ct <= 1:
        return T
    block_size = def_max_hp_p1 * def_max_hp_p1
    needed = att_max_fig_ct * block_size
    if needed > len(T):
        T_new = np.zeros(needed)
        T_new[:len(T)] = T
        T = T_new

    for att_fig_ct in range(2, att_max_fig_ct + 1):
        base_write_offset = (att_fig_ct - 1) * block_size
        for idx in range(def_max_hp_p1):
            write_offset = base_write_offset + idx * def_max_hp_p1
            read_offset_nm1 = (att_fig_ct - 2) * block_size + idx * def_max_hp_p1
            for idx2 in range(idx + 1):
                cur_prob = 0.0
                for idx3 in range(idx2, idx + 1):
                    cur_prob += T[read_offset_nm1 + idx3] * T[idx3 * def_max_hp_p1 + idx2]
                T[write_offset + idx2] = cur_prob
            for idx2 in range(idx + 1, def_max_hp_p1):
                T[write_offset + idx2] = 0.0
    return T


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
    T = make_grand_transition_table(att_max_fig_ct, def_max_hp + 1, T)
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
# Additional helper functions ported from JS
# ---------------------------------------------------------------------------

def repeat_melee_attack(def_max_hp_p1: int, T: np.ndarray) -> None:
    """
    M^2 = M * M.  Squares the transition table in-place (for haste double-attack).

    JS: repeat_melee_attack(def_max_hp_p1, tmp_transition_table, transition_table)
    """
    def_max_hp_p1_sq = def_max_hp_p1 * def_max_hp_p1
    tmp = np.zeros(def_max_hp_p1_sq)
    for idx in range(def_max_hp_p1):
        write_offset = idx * def_max_hp_p1
        read_offset_nm1 = idx * def_max_hp_p1
        for idx2 in range(idx + 1):
            cur_prob = 0.0
            for idx3 in range(idx2, idx + 1):
                cur_prob += T[read_offset_nm1 + idx3] * T[idx3 * def_max_hp_p1 + idx2]
            tmp[write_offset + idx2] = cur_prob
        for idx2 in range(idx + 1, def_max_hp_p1):
            tmp[write_offset + idx2] = 0.0
    T[:def_max_hp_p1_sq] = tmp[:def_max_hp_p1_sq]


def apply_touch_to_grand_transition_table(
    poison_strength: int,
    att_max_fig_ct: int,
    att_to_hit: float,
    def_max_fig_ct: int,
    def_fig_hp: int,
    grand_transition_table: np.ndarray,
) -> None:
    """
    Apply poison touch / stoning touch / dispel evil / destruction to the
    grand transition table.

    If poison_strength == 0, kills entire defending figures at a time
    (stoning touch/dispel evil).

    JS: apply_touch_to_grand_transition_table(poison_strength, att_max_fig_ct,
        att_to_hit, def_max_fig_ct, def_fig_hp, tmp_transition_table,
        grand_transition_table)
    """
    def_max_hp_p1 = def_max_fig_ct * def_fig_hp + 1
    stride = 1
    if poison_strength == 0:
        stride = def_fig_hp

    for transition_table_idx in range(att_max_fig_ct):
        grand_offset = def_max_hp_p1 * def_max_hp_p1 * transition_table_idx
        tmp = np.zeros(def_max_hp_p1 * def_max_hp_p1)

        if poison_strength > 0:
            max_damage = poison_strength * (transition_table_idx + 1)
            att_arr = binom_arr(1.0 - att_to_hit, max_damage)
        else:
            att_arr = binom_arr_stride(1.0 - att_to_hit, transition_table_idx + 1, def_fig_hp)
            max_damage = def_fig_hp * (transition_table_idx + 1)

        for def_start_hp in range(def_max_hp_p1):
            write_row_offset = def_start_hp * def_max_hp_p1
            row_offset = grand_offset + write_row_offset
            for def_mid_hp in range(def_start_hp + 1):
                cur_damage = 0
                while cur_damage <= max_damage:
                    cur_prob = grand_transition_table[row_offset + def_mid_hp] * att_arr[cur_damage]
                    if cur_damage < def_mid_hp:
                        tmp[write_row_offset + def_mid_hp - cur_damage] += cur_prob
                    else:
                        tmp[write_row_offset] += cur_prob
                    cur_damage += stride

        size = def_max_hp_p1 * def_max_hp_p1
        grand_transition_table[grand_offset:grand_offset + size] = tmp[:size]


def apply_immolation_to_grand_transition_table(
    att_immo: int,
    att_max_fig_ct: int,
    shields_vsimmo: int,
    def_toblock: float,
    def_max_fig_ct: int,
    def_fig_hp: int,
    def_invuln: int,
    grand_transition_table: np.ndarray,
) -> None:
    """
    Apply immolation damage to a grand transition table.

    Immolation power depends on number of defending, not attacking, figures.

    JS: apply_immolation_to_grand_transition_table(att_immo, att_max_fig_ct,
        shields_vsimmo, def_toblock, def_max_fig_ct, def_fig_hp, def_invuln,
        tmp_transition_table, immo_damage_array, grand_transition_table)
    """
    def_max_hp = def_max_fig_ct * def_fig_hp
    def_max_hp_p1 = def_max_hp + 1

    att_arr = binom_arr(0.7, att_immo)
    block_arr = get_block_arr(1.0 - def_toblock, shields_vsimmo, def_invuln)
    att_arr = apply_block(att_arr, block_arr)

    immo_single_fig_max = att_immo
    if att_immo > def_fig_hp:
        immo_single_fig_max = def_fig_hp
        for i in range(def_fig_hp + 1, att_immo + 1):
            if i < len(att_arr):
                att_arr[def_fig_hp] += att_arr[i]

    # immo_damage_array[x] = damage distribution for n-1 full figures
    # immo_damage_array[k * def_max_hp_p1 + x] = truncated distribution for
    #   single figure with k HP
    immo_damage_array = np.zeros(def_max_hp_p1 * (def_fig_hp + 1))
    immo_damage_array[0] = 1.0

    for i in range(1, def_fig_hp + 1):
        for i2 in range(i + 1):
            if i2 <= att_immo and i2 < len(att_arr):
                immo_damage_array[i * def_max_hp_p1 + i2] = att_arr[i2]
            else:
                immo_damage_array[i * def_max_hp_p1 + i2] = 0.0
        if i < att_immo:
            for i2 in range(i + 1, immo_single_fig_max + 1):
                if i2 < len(att_arr):
                    immo_damage_array[i * def_max_hp_p1 + i] += att_arr[i2]

    for transition_table_idx in range(att_max_fig_ct):
        grand_offset = def_max_hp_p1 * def_max_hp_p1 * transition_table_idx
        tmp = np.zeros(def_max_hp_p1 * def_max_hp_p1)

        for def_fig_ct in range(1, def_max_fig_ct + 1):
            if def_fig_ct > 1:
                # Backwards in-place update of immo_damage_array
                idx = immo_single_fig_max * (def_fig_ct - 1)
                if idx > def_max_hp:
                    idx = def_max_hp
                while idx >= 0:
                    cur_prob = 0.0
                    for i2 in range(immo_single_fig_max + 1):
                        if i2 < len(att_arr):
                            cur_prob += immo_damage_array[idx - i2] * att_arr[i2]
                        if i2 == idx:
                            break
                    immo_damage_array[idx] = cur_prob
                    idx -= 1

            for top_def_fig_hp in range(1, def_fig_hp + 1):
                cur_def_total_hp = (def_fig_ct - 1) * def_fig_hp + top_def_fig_hp
                write_row_offset = cur_def_total_hp * def_max_hp_p1
                row_offset = grand_offset + write_row_offset
                if top_def_fig_hp < immo_single_fig_max:
                    max_immo_dmg = top_def_fig_hp
                else:
                    max_immo_dmg = immo_single_fig_max
                max_immo_dmg += immo_single_fig_max * (def_fig_ct - 1)

                for i in range(max_immo_dmg + 1):
                    cur_prob = 0.0
                    max_top_fig_immo_dmg = top_def_fig_hp
                    if max_top_fig_immo_dmg > i:
                        max_top_fig_immo_dmg = i
                    for i2 in range(max_top_fig_immo_dmg + 1):
                        cur_prob += (immo_damage_array[i - i2] *
                                     immo_damage_array[top_def_fig_hp * def_max_hp_p1 + i2])
                    for i2 in range(cur_def_total_hp + 1):
                        if i2 <= i:
                            i3 = 0
                        else:
                            i3 = i2 - i
                        tmp[write_row_offset + i3] += cur_prob * grand_transition_table[row_offset + i2]

        size = def_max_hp_p1 * def_max_hp_p1
        grand_transition_table[grand_offset:grand_offset + size] = tmp[:size]


def apply_gazes(
    att_side: int,
    att_dgaze: int,
    att_sgaze: int,
    att_doomgaze: int,
    att_immo: int,
    att_max_hp: int,
    def_max_fig_ct: int,
    def_resist: int,
    def_bless: int,
    def_chaosnature: int,
    shields_vsimmo: int,
    def_toblock: float,
    def_fig_hp: int,
    def_invuln: int,
    joint_hparr: np.ndarray,
) -> None:
    """
    Apply gaze attacks (death gaze, stoning gaze, doom gaze + immolation)
    to the joint HP array.

    JS: apply_gazes(att_side, att_dgaze, att_sgaze, att_doomgaze, att_immo,
        att_max_hp, def_max_fig_ct, def_resist, def_bless, def_chaosnature,
        shields_vsimmo, def_toblock, def_fig_hp, def_invuln,
        transition_table, def_tmp_prob_array, workspace3, workspace4,
        joint_hparr)
    """
    def_max_hp = def_max_fig_ct * def_fig_hp
    T = construct_gaze_transition_table(
        def_max_fig_ct, def_resist, def_bless, def_chaosnature,
        def_fig_hp, att_dgaze, att_sgaze, att_doomgaze
    )
    if att_immo > 0:
        apply_immolation_to_grand_transition_table(
            att_immo, 1, shields_vsimmo, def_toblock,
            def_max_fig_ct, def_fig_hp, def_invuln, T
        )
    apply_grand_transition_table_to_side(
        att_side, 1, att_max_hp, def_max_hp, T, joint_hparr
    )


def simultaneous_resolve(
    max_fig0: int,
    max_fighp0: int,
    max_fig1: int,
    max_fighp1: int,
    grand_transition_table0: np.ndarray,
    grand_transition_table1: np.ndarray,
    joint_hparr: np.ndarray,
) -> None:
    """
    Resolve simultaneous melee attacks by both sides.

    For each joint (hp_a, hp_b) state, applies both grand transition tables
    simultaneously, producing the full joint distribution of outcomes.

    JS: simultaneous_resolve(max_fig0, max_fighp0, max_fig1, max_fighp1,
        grand_transition_table0, grand_transition_table1,
        tmp_hparr, joint_hparr)
    """
    maxhp0_p1 = 1 + max_fig0 * max_fighp0
    maxhp1_p1 = 1 + max_fig1 * max_fighp1
    hparr_size = maxhp0_p1 * maxhp1_p1
    tmp_hparr = np.zeros(hparr_size)

    for idx0a in range(maxhp0_p1):
        row_offset = idx0a * maxhp1_p1
        row_offset2 = idx0a * maxhp0_p1
        figs0_m1 = 0
        if idx0a > 0:
            figs0_m1 = math.ceil((idx0a - 0.5) / max_fighp0) - 1

        for idx1a in range(maxhp1_p1):
            pconst = joint_hparr[row_offset + idx1a]
            if pconst == 0:
                continue

            if idx0a == 0 or idx1a == 0:
                tmp_hparr[row_offset + idx1a] += pconst
            else:
                figs1_m1 = 0
                if idx1a > max_fighp1:
                    figs1_m1 = math.ceil((idx1a - 0.5) / max_fighp1) - 1

                row_offset2x = row_offset2 + maxhp0_p1 * maxhp0_p1 * figs1_m1
                row_offset3 = idx1a * maxhp1_p1 + maxhp1_p1 * maxhp1_p1 * figs0_m1

                for idx0b in range(idx0a + 1):
                    row_offset4 = idx0b * maxhp1_p1
                    pconst2 = pconst * grand_transition_table0[row_offset2x + idx0b]
                    for idx1b in range(idx1a + 1):
                        tmp_hparr[row_offset4 + idx1b] += (
                            pconst2 * grand_transition_table1[row_offset3 + idx1b]
                        )

    joint_hparr[:hparr_size] = tmp_hparr[:hparr_size]


# ---------------------------------------------------------------------------
# Melee functions — ported from JS lines 2289-2351
# ---------------------------------------------------------------------------

def calc_melee_grand_transition_table(
    att_side: int,
    maxfig: list[int],
    attack: list[int],
    shields: list[int],
    shields_vsimmo: list[int],
    resist: list[int],
    tohit: list[int],
    toblock: list[int],
    toblock_melee: list[int],
    max_fighp: list[int],
    poison: list[int],
    pois_resist_penalty: list[int],
    stouch: list[int],
    chaosnature_def: list[int],
    redtouch: list[int],
    bless: list[int],
    blacktouch: list[int],
    dispelevil: list[int],
    immo: list[int],
    color: list[int],
    undead: list[int],
    invuln: list[int],
    invuln_melee: list[int],
    iter_ct: int,
) -> np.ndarray:
    """
    Build the grand transition table for one side's melee attack.

    JS: calc_melee_grand_transition_table(att_side, maxfig, attack, shields,
        shields_vsimmo, resist, tohit, toblock, toblock_melee, max_fighp,
        poison, pois_resist_penalty, stouch, chaosnature_def, redtouch,
        bless, blacktouch, dispelevil, immo, color, undead, invuln,
        invuln_melee, iter_ct, grand_transition_table, workspace3, workspace4)
    """
    def_side = 1 - att_side
    def_max_hp = maxfig[def_side] * max_fighp[def_side]
    def_max_hp_p1 = def_max_hp + 1
    block_arr = get_block_arr(
        1.0 - (toblock_melee[def_side] * 0.01),
        shields[def_side],
        invuln_melee[def_side]
    )

    T = calc_transition_table(
        attack[att_side],
        tohit[att_side] * 0.01,
        block_arr,
        max_fighp[def_side],
        def_max_hp,
        0
    )

    if iter_ct == 2:
        repeat_melee_attack(def_max_hp_p1, T)

    T = make_grand_transition_table(maxfig[att_side], def_max_hp_p1, T)

    if attack[att_side] < 1:
        return T

    for _iter in range(iter_ct):
        # Poison touch
        if poison[att_side] > 0 and (resist[def_side] - pois_resist_penalty[def_side] < 10):
            apply_touch_to_grand_transition_table(
                poison[att_side],
                maxfig[att_side],
                1.0 - ((resist[def_side] - pois_resist_penalty[def_side]) * 0.1),
                maxfig[def_side],
                max_fighp[def_side],
                T
            )
        # Stoning touch
        if stouch[att_side] > 0 and (resist[def_side] + chaosnature_def[def_side] < stouch[att_side]):
            apply_touch_to_grand_transition_table(
                0,
                maxfig[att_side],
                (stouch[att_side] - resist[def_side] - chaosnature_def[def_side]) * 0.1,
                maxfig[def_side],
                max_fighp[def_side],
                T
            )
        # Destruction (red touch)
        if redtouch[att_side] > 0 and (resist[def_side] + chaosnature_def[def_side] + 3 * bless[def_side] < redtouch[att_side]):
            apply_touch_to_grand_transition_table(
                0,
                maxfig[att_side],
                (redtouch[att_side] - resist[def_side] - chaosnature_def[def_side] - 3 * bless[def_side]) * 0.1,
                maxfig[def_side],
                max_fighp[def_side],
                T
            )
        # Death touch (black touch)
        if blacktouch[att_side] > 0 and (resist[def_side] + 3 * bless[def_side] < blacktouch[att_side]):
            apply_touch_to_grand_transition_table(
                0,
                maxfig[att_side],
                (blacktouch[att_side] - resist[def_side] - 3 * bless[def_side]) * 0.1,
                maxfig[def_side],
                max_fighp[def_side],
                T
            )
        # Dispel Evil
        if dispelevil[att_side] and (color[def_side] == 2 or color[def_side] == 3):
            if undead[def_side]:
                tmp = 19
            else:
                tmp = 14
            if tmp > resist[def_side]:
                apply_touch_to_grand_transition_table(
                    0,
                    maxfig[att_side],
                    (tmp - resist[def_side]) * 0.1,
                    maxfig[def_side],
                    max_fighp[def_side],
                    T
                )
        # Immolation
        if immo[att_side] > 0:
            apply_immolation_to_grand_transition_table(
                immo[att_side],
                maxfig[att_side],
                shields_vsimmo[def_side],
                toblock[def_side] * 0.01,
                maxfig[def_side],
                max_fighp[def_side],
                invuln[def_side],
                T
            )

    return T


def calc_melee_one_side(
    att_side: int,
    maxfig: list[int],
    attack: list[int],
    shields: list[int],
    shields_vsimmo: list[int],
    resist: list[int],
    tohit: list[int],
    toblock: list[int],
    toblock_melee: list[int],
    max_fighp: list[int],
    poison: list[int],
    pois_resist_penalty: list[int],
    stouch: list[int],
    chaosnature_def: list[int],
    redtouch_melee: list[int],
    bless: list[int],
    blacktouch_melee: list[int],
    dispelevil: list[int],
    immo: list[int],
    color: list[int],
    undead: list[int],
    invuln: list[int],
    invuln_melee: list[int],
    joint_hparr: np.ndarray,
) -> None:
    """
    One side attacks in melee (non-simultaneous). Used for first-strike resolution.

    JS: calc_melee_one_side(att_side, maxfig, attack, shields, shields_vsimmo,
        resist, tohit, toblock, toblock_melee, max_fighp, poison,
        pois_resist_penalty, stouch, chaosnature_def, redtouch_melee, bless,
        blacktouch_melee, dispelevil, immo, color, undead, invuln,
        invuln_melee, grand_transition_table, tmp_hparr, workspace3,
        workspace4, joint_hparr)
    """
    def_side = 1 - att_side
    T = calc_melee_grand_transition_table(
        att_side, maxfig, attack, shields, shields_vsimmo, resist,
        tohit, toblock, toblock_melee, max_fighp, poison,
        pois_resist_penalty, stouch, chaosnature_def, redtouch_melee,
        bless, blacktouch_melee, dispelevil, immo, color, undead,
        invuln, invuln_melee, 1
    )
    apply_grand_transition_table_to_side(
        att_side, maxfig[att_side], max_fighp[att_side],
        maxfig[def_side] * max_fighp[def_side], T, joint_hparr
    )


def calc_melee_main(
    maxfig: list[int],
    attack: list[int],
    shields: list[int],
    shields_vsimmo: list[int],
    resist: list[int],
    tohit: list[int],
    toblock: list[int],
    toblock_melee: list[int],
    max_fighp: list[int],
    poison: list[int],
    pois_resist_penalty: list[int],
    stouch: list[int],
    chaosnature_def: list[int],
    redtouch_melee: list[int],
    bless: list[int],
    blacktouch_melee: list[int],
    dispelevil: list[int],
    immo: list[int],
    color: list[int],
    undead: list[int],
    invuln: list[int],
    invuln_melee: list[int],
    melee_iter_cts: list[int],
    joint_hparr: np.ndarray,
) -> None:
    """
    Simultaneous melee resolution. Both sides attack at the same time.

    JS: calc_melee_main(maxfig, attack, shields, shields_vsimmo, resist,
        tohit, toblock, toblock_melee, max_fighp, poison,
        pois_resist_penalty, stouch, chaosnature_def, redtouch_melee,
        bless, blacktouch_melee, dispelevil, immo, color, undead, invuln,
        invuln_melee, melee_iter_cts, grand_transition_table0,
        grand_transition_table1, tmp_hparr, workspace3, workspace4,
        joint_hparr)
    """
    # Side 0 attacks side 1: builds grand_transition_table1
    T1 = calc_melee_grand_transition_table(
        0, maxfig, attack, shields, shields_vsimmo, resist,
        tohit, toblock, toblock_melee, max_fighp, poison,
        pois_resist_penalty, stouch, chaosnature_def, redtouch_melee,
        bless, blacktouch_melee, dispelevil, immo, color, undead,
        invuln, invuln_melee, melee_iter_cts[0]
    )
    # Side 1 attacks side 0: builds grand_transition_table0
    T0 = calc_melee_grand_transition_table(
        1, maxfig, attack, shields, shields_vsimmo, resist,
        tohit, toblock, toblock_melee, max_fighp, poison,
        pois_resist_penalty, stouch, chaosnature_def, redtouch_melee,
        bless, blacktouch_melee, dispelevil, immo, color, undead,
        invuln, invuln_melee, melee_iter_cts[1]
    )
    simultaneous_resolve(
        maxfig[0], max_fighp[0],
        maxfig[1], max_fighp[1],
        T0, T1, joint_hparr
    )


# ---------------------------------------------------------------------------
# CombatState dataclass — mirrors the JS `mo` object
# ---------------------------------------------------------------------------


def _pair(default=0):
    """Helper: default factory for a 2-element list."""
    return field(default_factory=lambda: [default, default])


@dataclass
class CombatState:
    """
    All combat-relevant state for a two-sided engagement.
    Each field that is a list has index [0] for side A, [1] for side B.
    Mirrors the JavaScript `mo` object from AdvancedDamageCalculator.js.
    """
    # Core stats (per-figure values for multi-figure units)
    figures: list = field(default_factory=lambda: [1, 1])
    melee: list = field(default_factory=lambda: [0, 0])
    ranged: list = field(default_factory=lambda: [0, 0])
    rangedtype: list = field(default_factory=lambda: [0, 0])  # 0=none, 1=magic, 2=bow/sling, 3=rock, 4=thrown
    defense: list = field(default_factory=lambda: [0, 0])
    resist: list = field(default_factory=lambda: [0, 0])
    hp: list = field(default_factory=lambda: [1, 1])  # HP per figure
    breath: list = field(default_factory=lambda: [0, 0])
    breathtype: list = field(default_factory=lambda: [0, 0])  # 0=none, 1=thrown, 2=fire, 3=lightning

    # To-hit / to-block bonuses (in percentage points beyond base 30%)
    tohit_melee: list = field(default_factory=lambda: [0, 0])
    tohit_ranged: list = field(default_factory=lambda: [0, 0])
    tohit_breath: list = field(default_factory=lambda: [0, 0])
    toblock: list = field(default_factory=lambda: [0, 0])

    # Abilities
    lucky: list = field(default_factory=lambda: [0, 0])
    holybonus: list = field(default_factory=lambda: [0, 0])
    resistall: list = field(default_factory=lambda: [0, 0])
    bless: list = field(default_factory=lambda: [0, 0])

    # Immunities
    magimm: list = field(default_factory=lambda: [0, 0])
    weapimm: list = field(default_factory=lambda: [0, 0])
    missimm: list = field(default_factory=lambda: [0, 0])
    fireimm: list = field(default_factory=lambda: [0, 0])
    coldimm: list = field(default_factory=lambda: [0, 0])
    illimm: list = field(default_factory=lambda: [0, 0])
    deathimm: list = field(default_factory=lambda: [0, 0])
    poisimm: list = field(default_factory=lambda: [0, 0])
    stonimm: list = field(default_factory=lambda: [0, 0])

    # Touch attacks — per-attack-type
    poison: list = field(default_factory=lambda: [0, 0])
    pois_resist_penalty: list = field(default_factory=lambda: [0, 0])
    stouch_melee: list = field(default_factory=lambda: [0, 0])
    stouch_ranged: list = field(default_factory=lambda: [0, 0])
    stouch_breath: list = field(default_factory=lambda: [0, 0])
    redtouch_melee: list = field(default_factory=lambda: [0, 0])
    redtouch_ranged: list = field(default_factory=lambda: [0, 0])
    redtouch_breath: list = field(default_factory=lambda: [0, 0])
    blacktouch_melee: list = field(default_factory=lambda: [0, 0])
    blacktouch_ranged: list = field(default_factory=lambda: [0, 0])
    blacktouch_breath: list = field(default_factory=lambda: [0, 0])
    dispelevil_melee: list = field(default_factory=lambda: [0, 0])
    dispelevil_breath: list = field(default_factory=lambda: [0, 0])

    # Life steal
    lifesteal_melee: list = field(default_factory=lambda: [0, 0])
    lifesteal_ranged: list = field(default_factory=lambda: [0, 0])
    lifesteal_breath: list = field(default_factory=lambda: [0, 0])

    # Gaze attacks
    dgaze: list = field(default_factory=lambda: [0, 0])
    sgaze: list = field(default_factory=lambda: [0, 0])
    doomgaze: list = field(default_factory=lambda: [0, 0])
    gazeranged: list = field(default_factory=lambda: [0, 0])

    # Immolation
    immo: list = field(default_factory=lambda: [0, 0])

    # Special abilities
    illusion_melee: list = field(default_factory=lambda: [0, 0])
    illusion_ranged: list = field(default_factory=lambda: [0, 0])
    illusion_breath: list = field(default_factory=lambda: [0, 0])
    invis: list = field(default_factory=lambda: [0, 0])
    invuln: list = field(default_factory=lambda: [0, 0])
    righteous_base: list = field(default_factory=lambda: [0, 0])
    undead: list = field(default_factory=lambda: [0, 0])
    noncorp: list = field(default_factory=lambda: [0, 0])
    flying: list = field(default_factory=lambda: [0, 0])

    # Unit color: 0=none, 1=life, 2=death, 3=chaos, 4=nature, 5=sorcery
    color: list = field(default_factory=lambda: [0, 0])

    # Weapon type: 0=normal, 1=magic, 2=mithril, 3=adamantium
    wtype: list = field(default_factory=lambda: [0, 0])
    wmagic: list = field(default_factory=lambda: [0, 0])

    # Armor piercing, first strike, haste
    ap_melee: list = field(default_factory=lambda: [0, 0])
    ap_ranged: list = field(default_factory=lambda: [0, 0])
    ap_breath: list = field(default_factory=lambda: [0, 0])
    fs: list = field(default_factory=lambda: [0, 0])
    negatefs: list = field(default_factory=lambda: [0, 0])
    haste: list = field(default_factory=lambda: [0, 0])

    # Misc
    lshield: list = field(default_factory=lambda: [0, 0])
    lrange: list = field(default_factory=lambda: [0, 0])
    lheart: list = field(default_factory=lambda: [0, 0])
    regen: list = field(default_factory=lambda: [0, 0])
    level: list = field(default_factory=lambda: [0, 0])
    dt: list = field(default_factory=lambda: [0, 0])  # damage taken at start
    chaosnature_def: list = field(default_factory=lambda: [0, 0])

    # Chaos channels
    chaos_melee: list = field(default_factory=lambda: [0, 0])
    chaos_ranged: list = field(default_factory=lambda: [0, 0])
    chaos_breath: list = field(default_factory=lambda: [0, 0])

    # Enchantments (boolean flags as 0/1)
    e_bless: list = field(default_factory=lambda: [0, 0])
    e_holyarmor: list = field(default_factory=lambda: [0, 0])
    e_invuln: list = field(default_factory=lambda: [0, 0])
    e_righteous: list = field(default_factory=lambda: [0, 0])
    e_wraithform: list = field(default_factory=lambda: [0, 0])
    e_holyweap: list = field(default_factory=lambda: [0, 0])
    e_eldritch: list = field(default_factory=lambda: [0, 0])
    e_truesight: list = field(default_factory=lambda: [0, 0])
    e_truelight: list = field(default_factory=lambda: [0, 0])
    e_darkness: list = field(default_factory=lambda: [0, 0])
    e_heavlight: list = field(default_factory=lambda: [0, 0])
    e_charmlife: list = field(default_factory=lambda: [0, 0])
    e_cloudshadow: list = field(default_factory=lambda: [0, 0])

    # Global settings
    vnum: int = 0  # 0 = v1.31, 1 = v1.40n
    aura: int = 0  # 0=none, 1=chaos, 2=nature (color of node aura)


# ---------------------------------------------------------------------------
# Build CombatState from Unit objects
# ---------------------------------------------------------------------------

def _realm_to_color(realm: str) -> int:
    """Map realm string to color code."""
    realm_map = {
        "life": 1,
        "death": 2,
        "chaos": 3,
        "nature": 4,
        "sorcery": 5,
    }
    return realm_map.get(realm.lower(), 0)


def build_combat_state(unit_a, unit_b, vnum: int = 1) -> CombatState:
    """
    Construct a CombatState from two Unit objects (from battle.py).

    Maps Unit stats/abilities to the mo fields that calcround reads.
    Default vnum=1 (v1.40n rules).
    """
    units = [unit_a, unit_b]
    mo = CombatState()
    mo.vnum = vnum

    for side in range(2):
        u = units[side]
        mo.figures[side] = u.figures
        mo.melee[side] = u.melee
        mo.ranged[side] = u.ranged
        mo.rangedtype[side] = u.ranged_type
        mo.defense[side] = u.defense
        mo.resist[side] = u.resist
        mo.hp[side] = u.hp_per_figure
        mo.breath[side] = u.breath
        mo.breathtype[side] = u.breath_type
        mo.tohit_melee[side] = u.tohit_bonus
        mo.tohit_ranged[side] = u.tohit_bonus
        mo.tohit_breath[side] = u.tohit_bonus
        mo.toblock[side] = 0  # base block bonus beyond 30%
        mo.poison[side] = u.poison
        mo.immo[side] = u.immolation
        mo.lifesteal_melee[side] = u.life_steal
        mo.lifesteal_ranged[side] = u.life_steal
        mo.lifesteal_breath[side] = u.life_steal

        # Boolean abilities mapped to 0/1
        mo.fs[side] = 1 if u.first_strike else 0
        mo.negatefs[side] = 1 if u.negate_first_strike else 0
        mo.lshield[side] = 1 if u.large_shield else 0
        mo.magimm[side] = 1 if u.magic_immunity else 0
        mo.weapimm[side] = 1 if u.weapon_immunity else 0
        mo.missimm[side] = 1 if u.missile_immunity else 0
        mo.illimm[side] = 0  # set from unit data if available
        mo.ap_melee[side] = 1 if u.armor_piercing else 0
        mo.ap_ranged[side] = 1 if u.armor_piercing else 0
        mo.illusion_melee[side] = 1 if u.illusionary else 0
        mo.illusion_ranged[side] = 1 if u.illusionary else 0
        mo.illusion_breath[side] = 1 if u.illusionary else 0

        # Color from realm
        mo.color[side] = _realm_to_color(u.realm)

        # Magic creatures (fantastic units) have magic weapons
        if u.category in ("fantastic", "hero", "champion"):
            mo.wmagic[side] = 1

    # Apply cross-side immunities (same as JS calcdamage preprocessing)
    for side in range(2):
        if mo.negatefs[side]:
            mo.fs[1 - side] = 0
        if mo.poisimm[side]:
            mo.poison[1 - side] = 0
        if mo.deathimm[side]:
            mo.dgaze[1 - side] = 0
            mo.blacktouch_melee[1 - side] = 0
            mo.blacktouch_ranged[1 - side] = 0
            mo.blacktouch_breath[1 - side] = 0
            mo.lifesteal_melee[1 - side] = 0
            mo.lifesteal_ranged[1 - side] = 0
            mo.lifesteal_breath[1 - side] = 0
        if mo.stonimm[side]:
            mo.sgaze[1 - side] = 0
            mo.stouch_melee[1 - side] = 0
            mo.stouch_ranged[1 - side] = 0
            mo.stouch_breath[1 - side] = 0

    return mo


# ---------------------------------------------------------------------------
# calcround — the main combat round orchestrator
# Ported faithfully from JS lines 2352-2768
# ---------------------------------------------------------------------------

def calcround(
    att_side: int,
    ranged_dist: int,
    counterattack_penalty: int,
    joint_hparr: np.ndarray,
    mo: CombatState,
) -> None:
    """
    Calculate one combat round, applying all phases to the joint HP array.

    att_side: 0 = side A attacks, 1 = side B attacks
    ranged_dist: 0 for melee, 1 + floor(distance/3) for ranged
    counterattack_penalty: tohit penalty for the defender's counterattack
    joint_hparr: joint probability array (modified in place)
    mo: CombatState with all unit stats

    JS: calcround(att_side, ranged_dist, counterattack_penalty, joint_hparr,
        workspace, workspace2, workspace2b, workspace3, workspace4, mo)
    """
    def_side = 1 - att_side
    attack = [mo.melee[0], mo.melee[1]]
    ranged_attack = [mo.ranged[0], mo.ranged[1]]
    tohit_melee = [
        30 + mo.tohit_melee[0] + 10 * mo.lucky[0],
        30 + mo.tohit_melee[1] + 10 * mo.lucky[1],
    ]
    tohit_ranged = [
        30 + mo.tohit_ranged[0] + 10 * mo.lucky[0],
        30 + mo.tohit_ranged[1] + 10 * mo.lucky[1],
    ]
    tohit_breath = [
        30 + mo.tohit_breath[0] + 10 * mo.lucky[0],
        30 + mo.tohit_breath[1] + 10 * mo.lucky[1],
    ]
    resist = [
        mo.resist[0] + mo.lucky[0] + mo.holybonus[0] + mo.resistall[0],
        mo.resist[1] + mo.lucky[1] + mo.holybonus[1] + mo.resistall[1],
    ]
    breath = [mo.breath[0], mo.breath[1]]
    gazeranged = [mo.gazeranged[0], mo.gazeranged[1]]
    dgaze = [mo.dgaze[0], mo.dgaze[1]]
    sgaze = [mo.sgaze[0], mo.sgaze[1]]
    dispelevil_melee = [mo.dispelevil_melee[0], mo.dispelevil_melee[1]]
    dispelevil_breath = [mo.dispelevil_breath[0], mo.dispelevil_breath[1]]
    stouch_melee = [mo.stouch_melee[0], mo.stouch_melee[1]]
    stouch_ranged = [mo.stouch_ranged[0], mo.stouch_ranged[1]]
    stouch_breath = [mo.stouch_breath[0], mo.stouch_breath[1]]
    redtouch_melee = [mo.redtouch_melee[0], mo.redtouch_melee[1]]
    redtouch_ranged = [mo.redtouch_ranged[0], mo.redtouch_ranged[1]]
    redtouch_breath = [mo.redtouch_breath[0], mo.redtouch_breath[1]]
    blacktouch_melee = [mo.blacktouch_melee[0], mo.blacktouch_melee[1]]
    blacktouch_ranged = [mo.blacktouch_ranged[0], mo.blacktouch_ranged[1]]
    blacktouch_breath = [mo.blacktouch_breath[0], mo.blacktouch_breath[1]]
    invuln_melee = [mo.invuln[0], mo.invuln[1]]
    invuln_ranged = [0, 0]
    invuln_breath = [0, 0]
    righteous_base = [mo.righteous_base[0], mo.righteous_base[1]]
    lsteal_melee = [mo.lifesteal_melee[0], mo.lifesteal_melee[1]]
    lsteal_ranged = [mo.lifesteal_ranged[0], mo.lifesteal_ranged[1]]
    lsteal_breath = [mo.lifesteal_breath[0], mo.lifesteal_breath[1]]
    weapimm = [mo.weapimm[0], mo.weapimm[1]]
    bless_local = [mo.bless[0], mo.bless[1]]

    # v1.31 lucky bug: reduce opponent's to-hit
    if mo.vnum == 0:
        tohit_melee[0] -= 10 * mo.lucky[1]
        tohit_ranged[0] -= 10 * mo.lucky[1]
        tohit_breath[0] -= 10 * mo.lucky[1]
        tohit_melee[1] -= 10 * mo.lucky[0]
        tohit_ranged[1] -= 10 * mo.lucky[0]
        tohit_breath[1] -= 10 * mo.lucky[0]

    shields = [mo.defense[0] + mo.holybonus[0], mo.defense[1] + mo.holybonus[1]]
    shields_vsmelee = [0, 0]
    shields_vsranged = [0, 0]
    shields_vsimmo = [0, 0]
    shields_vsbreath = [0, 0]
    toblock = [
        30 + mo.toblock[0] + 10 * mo.lucky[0],
        30 + mo.toblock[1] + 10 * mo.lucky[1],
    ]
    toblock_melee = [toblock[0], toblock[1]]
    toblock_ranged = [toblock[0], toblock[1]]
    toblock_breath = [toblock[0], toblock[1]]
    melee_iter_cts = [1, 1]
    light_modifier = (mo.e_truelight[0] + mo.e_truelight[1]
                      - mo.e_darkness[0] - mo.e_darkness[1])

    # --- Enchantment preprocessing ---
    for side in range(2):
        if mo.e_bless[side]:
            bless_local[side] = 1
        if mo.e_holyarmor[side]:
            shields[side] += 2
        if mo.e_invuln[side]:
            invuln_melee[side] = 1
        invuln_ranged[side] = invuln_melee[side]
        invuln_breath[side] = invuln_melee[side]
        if mo.e_righteous[side]:
            righteous_base[side] = 1
        if mo.e_wraithform[side]:
            weapimm[side] = 1

    # --- Immunity / modifier logic ---
    for side in range(2):
        if attack[side] > 0:
            attack[side] += mo.holybonus[side]

        if mo.magimm[side] or righteous_base[side]:
            dgaze[1 - side] = 0
            redtouch_melee[1 - side] = 0
            redtouch_ranged[1 - side] = 0
            redtouch_breath[1 - side] = 0
            blacktouch_melee[1 - side] = 0
            blacktouch_ranged[1 - side] = 0
            blacktouch_breath[1 - side] = 0
            lsteal_melee[1 - side] = 0
            lsteal_ranged[1 - side] = 0
            lsteal_breath[1 - side] = 0
            if mo.magimm[side]:
                sgaze[1 - side] = 0
                dispelevil_melee[1 - side] = 0
                dispelevil_breath[1 - side] = 0
                stouch_melee[1 - side] = 0
                stouch_ranged[1 - side] = 0
                stouch_breath[1 - side] = 0

        # Weapon type bonuses
        if mo.wtype[side] > 1:
            shields[side] += mo.wtype[side] - 1
        if mo.wtype[side] > 0:
            tohit_melee[side] += 10
            if mo.rangedtype[side] > 3:
                tohit_ranged[side] += 10
            if ranged_dist == 0:
                if attack[side] > 0:
                    attack[side] += mo.wtype[side] - 1
                if breath[side] > 0 and mo.breathtype[side] == 1:
                    breath[side] += mo.wtype[side] - 1
            elif ranged_attack[side] > 0 and mo.rangedtype[side] > 3:
                ranged_attack[side] += mo.wtype[side] - 1

        # Holy Weapon enchantment
        if mo.e_holyweap[side]:
            tohit_melee[side] += 10
            if mo.rangedtype[side] > 3:
                tohit_ranged[side] += 10

        # True Light / Darkness modifier
        if light_modifier != 0 and (mo.color[side] == 1 or mo.color[side] == 2):
            delta = light_modifier
            if mo.color[side] == 2:
                delta = -delta
            attack[side] += delta
            if attack[side] < 0:
                attack[side] = 0
            if ranged_attack[side] > 0:
                ranged_attack[side] += delta
                if ranged_attack[side] < 0:
                    ranged_attack[side] = 0
            if gazeranged[side] > 0:
                gazeranged[side] += delta
                if gazeranged[side] < 0:
                    gazeranged[side] = 0
            if breath[side] > 0:
                breath[side] += delta
                if breath[side] < 0:
                    breath[side] = 0
            shields[side] += delta
            if shields[side] < 0:
                shields[side] = 0
            resist[side] += delta
            if resist[side] < 0:
                resist[side] = 0

        shields_vsmelee[side] = shields[side]
        shields_vsranged[side] = shields[side]
        shields_vsimmo[side] = shields[side]
        shields_vsbreath[side] = shields[side]

        # Large shield
        if mo.lshield[side]:
            shields_vsranged[side] += 2
            shields_vsimmo[side] += 2
            shields_vsbreath[side] += 2

        # Eldritch Weapon: reduces opponent's block chance
        if mo.e_eldritch[side]:
            toblock_melee[1 - side] -= 10
            if mo.rangedtype[side] == 4:
                toblock_ranged[1 - side] -= 10
            if mo.breathtype[side] == 1:
                toblock_breath[1 - side] -= 10

    # --- Bless / resist-elements / armor-piercing / weapon-immunity / etc. ---
    for side in range(2):
        weap_imm_applied = 0

        if bless_local[side]:
            if mo.color[1 - side] == 2 or mo.color[1 - side] == 3:
                shields_vsmelee[side] += 3
            if mo.rangedtype[1 - side] == 1:
                shields_vsranged[side] += 3
            if (mo.breathtype[1 - side] > 1 or
                (gazeranged[1 - side] > 0 and
                 (mo.color[1 - side] == 2 or mo.color[1 - side] == 3))):
                shields_vsbreath[side] += 3
            shields_vsimmo[side] += 3

        if mo.chaosnature_def[side]:
            if mo.rangedtype[1 - side] == 1 or mo.rangedtype[1 - side] == 2:
                shields_vsranged[side] += mo.chaosnature_def[side]
            if (mo.breathtype[1 - side] > 1 or
                (gazeranged[1 - side] > 0 and
                 (mo.color[1 - side] == 3 or mo.color[1 - side] == 4))):
                shields_vsbreath[side] += mo.chaosnature_def[side]
            shields_vsimmo[side] += mo.chaosnature_def[side]

        # Armor piercing
        if mo.ap_ranged[1 - side]:
            shields_vsranged[side] = shields_vsranged[side] // 2
        if mo.ap_breath[1 - side]:
            shields_vsbreath[side] = shields_vsbreath[side] // 2
        if mo.ap_melee[1 - side]:
            shields_vsmelee[side] = shields_vsmelee[side] // 2

        # Weapon immunity vs melee
        if (weapimm[side] and not mo.wmagic[1 - side] and
                not mo.e_holyweap[1 - side] and not mo.e_eldritch[1 - side] and
                ranged_dist == 0 and shields_vsmelee[side] < 10):
            if mo.vnum == 0:
                weap_imm_applied = 1
            shields_vsmelee[side] = 10

        # Fire/magic immunity vs immolation
        if mo.fireimm[side] or mo.magimm[side] or righteous_base[side]:
            shields_vsimmo[side] = 50

        # Defender-specific ranged immunities
        if side == def_side:
            if ranged_dist > 0:
                if mo.rangedtype[att_side] < 4:
                    if (mo.magimm[def_side] or
                        (mo.e_righteous[def_side] and mo.rangedtype[att_side] == 1)):
                        shields_vsranged[def_side] = 50
                else:
                    # Range penalty for thrown attacks
                    if ranged_dist > 1:
                        if mo.lrange[att_side]:
                            tohit_ranged[att_side] -= 10
                        else:
                            tohit_ranged[att_side] -= 10 * (ranged_dist - 1)
                    if mo.rangedtype[att_side] == 4:
                        if (weapimm[def_side] and not mo.wmagic[att_side] and
                                not mo.e_holyweap[1 - side] and shields[def_side] < 10):
                            if mo.vnum == 0:
                                weap_imm_applied = 1
                            shields_vsranged[def_side] = 10
                        if mo.missimm[def_side] and not weap_imm_applied:
                            shields_vsranged[def_side] = 50

        # Breath immunity
        if mo.breathtype[1 - side] > 1:
            if mo.magimm[side] or mo.e_righteous[side]:
                shields_vsbreath[side] = 50
            elif mo.breathtype[1 - side] == 2 and mo.fireimm[side]:
                shields_vsbreath[side] = 50

        # Gaze ranged immunity
        if gazeranged[1 - side] > 0:
            if (mo.magimm[side] or
                (mo.e_righteous[side] and
                 (mo.color[1 - side] == 2 or mo.color[1 - side] == 3))):
                shields_vsbreath[side] = 50

        # Illusion attacks bypass defense
        if not mo.illimm[side] and not mo.e_truesight[side]:
            if mo.illusion_melee[1 - side]:
                shields_vsmelee[side] = 0
            if mo.illusion_ranged[1 - side]:
                shields_vsranged[side] = 0
            if mo.illusion_breath[1 - side]:
                shields_vsbreath[side] = 0
            if mo.invis[1 - side]:
                tohit_melee[side] -= 10
                tohit_breath[side] -= 10

    # Counterattack penalty
    tohit_melee[def_side] -= counterattack_penalty

    # Clamp tohit values
    for side in range(2):
        tohit_melee[side] = max(10, min(100, tohit_melee[side]))
        tohit_ranged[side] = max(10, min(100, tohit_ranged[side]))
        tohit_breath[side] = max(10, min(100, tohit_breath[side]))

        # Chaos channels melee
        if mo.chaos_melee[side]:
            attack[side] = attack[side] // 2
            tohit_melee[side] = 100
            shields_vsmelee[1 - side] = 0
            invuln_melee[1 - side] = 0
        if mo.chaos_ranged[side]:
            ranged_attack[side] = ranged_attack[side] // 2
            tohit_ranged[side] = 100
            shields_vsranged[1 - side] = 0
            invuln_ranged[1 - side] = 0
        if mo.chaos_breath[side]:
            breath[side] = breath[side] // 2
            tohit_breath[side] = 100
            shields_vsbreath[1 - side] = 0
            invuln_breath[1 - side] = 0

    # ===== ATTACK PHASES =====

    if ranged_dist > 0:
        # --- Ranged phase ---
        iter_ct = 1
        if mo.haste[att_side] and mo.rangedtype[att_side] > 3:
            iter_ct = 2
        for _iter in range(iter_ct):
            calc_one_attack(
                att_side, mo.figures[att_side], ranged_attack[att_side],
                tohit_ranged[att_side] * 0.01, mo.hp[att_side],
                mo.figures[def_side], shields_vsranged[def_side],
                toblock_ranged[def_side] * 0.01, mo.hp[def_side],
                invuln_ranged[def_side], joint_hparr
            )
            if stouch_ranged[att_side] > resist[def_side] + mo.chaosnature_def[def_side]:
                calc_one_attack(
                    att_side, mo.figures[att_side], 1,
                    (stouch_ranged[att_side] - resist[def_side] - mo.chaosnature_def[def_side]) * 0.1,
                    mo.hp[att_side], mo.figures[def_side], 0, 0,
                    mo.hp[def_side], 0, joint_hparr, 1
                )
            if redtouch_ranged[att_side] > resist[def_side] + mo.chaosnature_def[def_side]:
                calc_one_attack(
                    att_side, mo.figures[att_side], 1,
                    (redtouch_ranged[att_side] - resist[def_side] - mo.chaosnature_def[def_side]) * 0.1,
                    mo.hp[att_side], mo.figures[def_side], 0, 0,
                    mo.hp[def_side], 0, joint_hparr, 1
                )
            if blacktouch_ranged[att_side] > resist[def_side] + 3 * bless_local[def_side]:
                calc_one_attack(
                    att_side, mo.figures[att_side], 1,
                    (blacktouch_ranged[att_side] - resist[def_side] - 3 * bless_local[def_side]) * 0.1,
                    mo.hp[att_side], mo.figures[def_side], 0, 0,
                    mo.hp[def_side], 0, joint_hparr, 1
                )
    else:
        # --- Melee combat ---

        # Breath phase (attacker only, on approach)
        if breath[att_side] > 0:
            iter_ct = 1 + mo.haste[att_side]
            for _iter in range(iter_ct):
                calc_one_attack(
                    att_side, mo.figures[att_side], breath[att_side],
                    tohit_breath[att_side] * 0.01, mo.hp[att_side],
                    mo.figures[def_side], shields_vsbreath[def_side],
                    toblock_breath[def_side] * 0.01, mo.hp[def_side],
                    invuln_breath[def_side], joint_hparr
                )
                if stouch_breath[att_side] > resist[def_side] + mo.chaosnature_def[def_side]:
                    calc_one_attack(
                        att_side, mo.figures[att_side], 1,
                        (stouch_breath[att_side] - resist[def_side] - mo.chaosnature_def[def_side]) * 0.1,
                        mo.hp[att_side], mo.figures[def_side], 0, 0,
                        mo.hp[def_side], 0, joint_hparr, 1
                    )
                if redtouch_breath[att_side] > resist[def_side] + mo.chaosnature_def[def_side]:
                    calc_one_attack(
                        att_side, mo.figures[att_side], 1,
                        (redtouch_breath[att_side] - resist[def_side] - mo.chaosnature_def[def_side]) * 0.1,
                        mo.hp[att_side], mo.figures[def_side], 0, 0,
                        mo.hp[def_side], 0, joint_hparr, 1
                    )
                if blacktouch_breath[att_side] > resist[def_side] + 3 * bless_local[def_side]:
                    calc_one_attack(
                        att_side, mo.figures[att_side], 1,
                        (blacktouch_breath[att_side] - resist[def_side] - 3 * bless_local[def_side]) * 0.1,
                        mo.hp[att_side], mo.figures[def_side], 0, 0,
                        mo.hp[def_side], 0, joint_hparr, 1
                    )
                if (dispelevil_breath[att_side] and
                        (mo.color[def_side] == 2 or mo.color[def_side] == 3)):
                    tmp = 19 if mo.undead[def_side] else 14
                    if tmp > resist[def_side]:
                        calc_one_attack(
                            att_side, mo.figures[att_side], 1,
                            (tmp - resist[def_side]) * 0.1,
                            mo.hp[att_side], mo.figures[def_side], 0, 0,
                            mo.hp[def_side], 0, joint_hparr, 1
                        )

        # Gaze phase — attacker
        if (gazeranged[att_side] > 0 and
                (dgaze[att_side] > resist[def_side] + 3 * bless_local[def_side] or
                 sgaze[att_side] > resist[def_side] + mo.chaosnature_def[def_side] or
                 mo.doomgaze[att_side] > 0)):
            apply_gazes(
                att_side, dgaze[att_side], sgaze[att_side],
                mo.doomgaze[att_side], mo.immo[att_side],
                mo.figures[att_side] * mo.hp[att_side],
                mo.figures[def_side], resist[def_side],
                bless_local[def_side], mo.chaosnature_def[def_side],
                shields_vsimmo[def_side], toblock[def_side] * 0.01,
                mo.hp[def_side], mo.invuln[def_side], joint_hparr
            )
            if not mo.doomgaze[att_side]:
                calc_one_attack(
                    att_side, mo.figures[att_side], gazeranged[att_side],
                    tohit_breath[att_side] * 0.01, mo.hp[att_side],
                    mo.figures[def_side], shields_vsbreath[def_side],
                    toblock[def_side] * 0.01, mo.hp[def_side],
                    invuln_breath[def_side], joint_hparr
                )

        # Gaze phase — defender
        if (gazeranged[def_side] > 0 and
                (dgaze[def_side] > resist[att_side] + 3 * bless_local[att_side] or
                 sgaze[def_side] > resist[att_side] + mo.chaosnature_def[att_side] or
                 mo.doomgaze[def_side] > 0)):
            apply_gazes(
                def_side, dgaze[def_side], sgaze[def_side],
                mo.doomgaze[def_side], mo.immo[def_side],
                mo.figures[def_side] * mo.hp[def_side],
                mo.figures[att_side], resist[att_side],
                bless_local[att_side], mo.chaosnature_def[att_side],
                shields_vsimmo[att_side], toblock[att_side] * 0.01,
                mo.hp[att_side], mo.invuln[def_side], joint_hparr
            )
            if not mo.doomgaze[def_side]:
                calc_one_attack(
                    def_side, mo.figures[def_side], gazeranged[def_side],
                    tohit_breath[def_side] * 0.01, mo.hp[def_side],
                    mo.figures[att_side], shields_vsbreath[att_side],
                    toblock[att_side] * 0.01, mo.hp[att_side],
                    invuln_breath[att_side], joint_hparr
                )

        # --- First Strike melee ---
        if mo.fs[att_side]:
            if lsteal_melee[0] <= resist[1] and lsteal_melee[1] <= resist[0]:
                calc_melee_one_side(
                    att_side, mo.figures, attack, shields_vsmelee,
                    shields_vsimmo, resist, tohit_melee, toblock,
                    toblock_melee, mo.hp, mo.poison, mo.pois_resist_penalty,
                    stouch_melee, mo.chaosnature_def, redtouch_melee,
                    bless_local, blacktouch_melee, dispelevil_melee,
                    mo.immo, mo.color, mo.undead, mo.invuln, invuln_melee,
                    joint_hparr
                )
                if not mo.haste[att_side]:
                    iter_ct = 1 + mo.haste[def_side]
                    for _iter in range(iter_ct):
                        calc_melee_one_side(
                            def_side, mo.figures, attack, shields_vsmelee,
                            shields_vsimmo, resist, tohit_melee, toblock,
                            toblock_melee, mo.hp, mo.poison,
                            mo.pois_resist_penalty, stouch_melee,
                            mo.chaosnature_def, redtouch_melee, bless_local,
                            blacktouch_melee, dispelevil_melee, mo.immo,
                            mo.color, mo.undead, mo.invuln, invuln_melee,
                            joint_hparr
                        )
            else:
                calc_melee_one_side(
                    att_side, mo.figures, attack, shields_vsmelee,
                    shields_vsimmo, resist, tohit_melee, toblock,
                    toblock_melee, mo.hp, mo.poison, mo.pois_resist_penalty,
                    stouch_melee, mo.chaosnature_def, redtouch_melee,
                    bless_local, blacktouch_melee, dispelevil_melee,
                    mo.immo, mo.color, mo.undead, mo.invuln, invuln_melee,
                    joint_hparr
                )
                if not mo.haste[att_side]:
                    iter_ct = 1 + mo.haste[def_side]
                    for _iter in range(iter_ct):
                        calc_melee_one_side(
                            def_side, mo.figures, attack, shields_vsmelee,
                            shields_vsimmo, resist, tohit_melee, toblock,
                            toblock_melee, mo.hp, mo.poison,
                            mo.pois_resist_penalty, stouch_melee,
                            mo.chaosnature_def, redtouch_melee, bless_local,
                            blacktouch_melee, dispelevil_melee, mo.immo,
                            mo.color, mo.undead, mo.invuln, invuln_melee,
                            joint_hparr
                        )

        # --- Normal (simultaneous) melee ---
        if not mo.fs[att_side] or mo.haste[att_side]:
            if mo.haste[att_side] and not mo.fs[att_side]:
                melee_iter_cts[att_side] = 2
            if mo.haste[def_side]:
                melee_iter_cts[def_side] = 2
            if lsteal_melee[0] <= resist[1] and lsteal_melee[1] <= resist[0]:
                calc_melee_main(
                    mo.figures, attack, shields_vsmelee, shields_vsimmo,
                    resist, tohit_melee, toblock, toblock_melee, mo.hp,
                    mo.poison, mo.pois_resist_penalty, stouch_melee,
                    mo.chaosnature_def, redtouch_melee, bless_local,
                    blacktouch_melee, dispelevil_melee, mo.immo, mo.color,
                    mo.undead, mo.invuln, invuln_melee, melee_iter_cts,
                    joint_hparr
                )
            else:
                calc_melee_main(
                    mo.figures, attack, shields_vsmelee, shields_vsimmo,
                    resist, tohit_melee, toblock, toblock_melee, mo.hp,
                    mo.poison, mo.pois_resist_penalty, stouch_melee,
                    mo.chaosnature_def, redtouch_melee, bless_local,
                    blacktouch_melee, dispelevil_melee, mo.immo, mo.color,
                    mo.undead, mo.invuln, invuln_melee, melee_iter_cts,
                    joint_hparr
                )


# ---------------------------------------------------------------------------
# run_combat — full combat resolution
# ---------------------------------------------------------------------------

def run_combat(unit_a, unit_b, mode: str = "melee", vnum: int = 1,
               max_rounds: int = 50) -> dict:
    """
    Run a full combat between two units using the wiki calculator's math.

    Parameters:
        unit_a, unit_b: Unit objects (from battle.py)
        mode: "melee" — melee engagement (ranged_dist=0), alternating attacker
              "ranged" — ranged attack round first, then melee rounds
        vnum: 0 = v1.31 rules, 1 = v1.40n rules
        max_rounds: safety cap on number of rounds

    Returns dict with:
        win_a: probability side A wins
        win_b: probability side B wins
        draw: probability of mutual kill
        avg_hp_a: expected HP remaining if A wins
        avg_hp_b: expected HP remaining if B wins
        rounds: number of rounds run
    """
    mo = build_combat_state(unit_a, unit_b, vnum=vnum)

    max_hp_a = mo.figures[0] * mo.hp[0]
    max_hp_b = mo.figures[1] * mo.hp[1]
    maxhp0_p1 = max_hp_a + 1
    maxhp1_p1 = max_hp_b + 1
    hparr_size = maxhp0_p1 * maxhp1_p1

    joint_hparr = np.zeros(hparr_size)
    # Start at full HP: both sides alive
    joint_hparr[max_hp_a * maxhp1_p1 + max_hp_b] = 1.0

    ranged_dist = 2 if mode == "ranged" else 0
    round_num = 0

    for round_num in range(1, max_rounds + 1):
        # Check how much probability is in "both alive" states
        undecided = 0.0
        for hp_a in range(1, maxhp0_p1):
            for hp_b in range(1, maxhp1_p1):
                undecided += joint_hparr[hp_a * maxhp1_p1 + hp_b]
        if undecided < 1e-10:
            break

        if round_num == 1 and ranged_dist > 0:
            # First round: ranged attack by side 0
            calcround(0, ranged_dist, 0, joint_hparr, mo)
        else:
            # Melee rounds: alternate who attacks
            # Round 1 (or 2 if ranged): side 0 attacks
            # Round 2 (or 3): side 1 attacks, etc.
            effective_round = round_num if ranged_dist == 0 else round_num - 1
            att_side = (effective_round - 1) % 2
            calcround(att_side, 0, 0, joint_hparr, mo)

    # Extract results
    win_a = 0.0
    win_b = 0.0
    draw = 0.0
    avg_hp_a = 0.0
    avg_hp_b = 0.0

    for hp_a in range(maxhp0_p1):
        for hp_b in range(maxhp1_p1):
            p = joint_hparr[hp_a * maxhp1_p1 + hp_b]
            if p == 0:
                continue
            if hp_a == 0 and hp_b == 0:
                draw += p
            elif hp_b == 0 and hp_a > 0:
                win_a += p
                avg_hp_a += p * hp_a
            elif hp_a == 0 and hp_b > 0:
                win_b += p
                avg_hp_b += p * hp_b

    if win_a > 0:
        avg_hp_a /= win_a
    if win_b > 0:
        avg_hp_b /= win_b

    return {
        "win_a": win_a,
        "win_b": win_b,
        "draw": draw,
        "undecided": 1.0 - win_a - win_b - draw,
        "avg_hp_a": avg_hp_a,
        "avg_hp_b": avg_hp_b,
        "rounds": round_num,
    }


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


def test_great_drake_vs_war_bears():
    """
    Test: Great Drake vs War Bears using the wiki calculator port.
    Great Drake: 1 fig, 30 melee, 10 def, 10 resist, 10 HP, fire breath 20
    War Bears: 4 fig, 7 melee, 4 def, 6 resist, 7 HP
    """
    # Import Unit from battle.py — but since we may not have it on the path,
    # define simple stand-in objects inline
    try:
        from battle import Unit
        drake = Unit(
            name="Great Drake",
            figures=1,
            hp_per_figure=10,
            melee=30,
            defense=10,
            resist=10,
            breath=20,
            breath_type=2,  # fire breath
            category="fantastic",
            realm="chaos",
        )
        bears = Unit(
            name="War Bears",
            figures=4,
            hp_per_figure=7,
            melee=7,
            defense=4,
            resist=6,
            category="normal",
            realm="nature",
        )
    except ImportError:
        # Minimal fallback using a simple namespace
        class SimpleUnit:
            def __init__(self, **kwargs):
                # Set defaults
                self.name = ""
                self.figures = 1
                self.hp_per_figure = 1
                self.melee = 0
                self.ranged = 0
                self.ranged_type = 0
                self.defense = 0
                self.resist = 0
                self.tohit_bonus = 0
                self.breath = 0
                self.breath_type = 0
                self.armor_piercing = False
                self.first_strike = False
                self.large_shield = False
                self.magic_immunity = False
                self.weapon_immunity = False
                self.missile_immunity = False
                self.illusionary = False
                self.negate_first_strike = False
                self.poison = 0
                self.life_steal = 0
                self.immolation = 0
                self.cost = 0
                self.category = ""
                self.race = ""
                self.realm = "none"
                for k, v in kwargs.items():
                    setattr(self, k, v)
                self.total_hp = self.figures * self.hp_per_figure

        drake = SimpleUnit(
            name="Great Drake",
            figures=1,
            hp_per_figure=10,
            melee=30,
            defense=10,
            resist=10,
            breath=20,
            breath_type=2,  # fire breath
            category="fantastic",
            realm="chaos",
        )
        bears = SimpleUnit(
            name="War Bears",
            figures=4,
            hp_per_figure=7,
            melee=7,
            defense=4,
            resist=6,
            category="normal",
            realm="nature",
        )

    print("=" * 60)
    print("Great Drake vs War Bears — melee engagement")
    print("=" * 60)

    result = run_combat(drake, bears, mode="melee")
    print(f"  Drake wins: {result['win_a']*100:.1f}%  (avg {result['avg_hp_a']:.1f} HP remaining)")
    print(f"  Bears win:  {result['win_b']*100:.1f}%  (avg {result['avg_hp_b']:.1f} HP remaining)")
    print(f"  Draw:       {result['draw']*100:.1f}%")
    print(f"  Undecided:  {result['undecided']*100:.1f}%")

    # Show marginal HP distributions
    mo = build_combat_state(drake, bears)
    max_hp_a = mo.figures[0] * mo.hp[0]
    max_hp_b = mo.figures[1] * mo.hp[1]
    maxhp1_p1 = max_hp_b + 1

    joint_hparr = np.zeros((max_hp_a + 1) * (max_hp_b + 1))
    joint_hparr[max_hp_a * maxhp1_p1 + max_hp_b] = 1.0
    calcround(0, 0, 0, joint_hparr, mo)

    print("\n  Drake HP distribution (marginal):")
    for hp in range(max_hp_a, -1, -1):
        prob = sum(joint_hparr[hp * maxhp1_p1 + hb] for hb in range(maxhp1_p1))
        if prob > 0.001:
            print(f"    HP {hp:2d}: {prob*100:6.2f}%")

    print("\n  Bears HP distribution (marginal):")
    for hp in range(max_hp_b, -1, -1):
        prob = sum(joint_hparr[ha * maxhp1_p1 + hp] for ha in range(max_hp_a + 1))
        if prob > 0.001:
            print(f"    HP {hp:2d}: {prob*100:6.2f}%")


if __name__ == "__main__":
    validate_against_js_example()
    print()
    test_great_drake_vs_war_bears()
