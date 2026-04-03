"""
Master of Magic Battle Simulator
Markov chain-based combat resolution.

Three layers:
  MarkovGenerator — builds transition matrices from unit stats
  BattleEngine    — resolves a single fight using the Markov matrix
  Arena           — runs matchups across groups of units, collects results
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data: Unit representation
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """Combat-relevant stats for a single unit type."""
    name: str
    figures: int
    hp_per_figure: int
    melee: int
    ranged: int = 0
    ranged_type: int = 0  # 0=none, 1=bow, 2=magic ranged
    defense: int = 0
    resist: int = 0
    tohit_bonus: int = 0  # percentage bonus (e.g., 30 means +30%)
    breath: int = 0
    breath_type: int = 0
    # Abilities that affect combat math
    armor_piercing: bool = False
    first_strike: bool = False
    large_shield: bool = False
    magic_immunity: bool = False
    weapon_immunity: bool = False
    missile_immunity: bool = False
    illusionary: bool = False
    negate_first_strike: bool = False
    poison: int = 0
    life_steal: int = 0
    immolation: int = 0
    # Metadata
    cost: int = 0
    category: str = ""
    race: str = ""
    realm: str = ""

    @property
    def total_hp(self) -> int:
        return self.figures * self.hp_per_figure

    @property
    def base_tohit(self) -> float:
        """Base to-hit chance as a fraction (0.0 to 1.0)."""
        return min((30 + self.tohit_bonus) / 100.0, 0.99)

    @classmethod
    def from_json(cls, data: dict) -> "Unit":
        """Load from unit-data.json format."""
        abilities = data.get("abilities", {})
        # Determine cost from the scraped data or calculator data
        cost = 0
        if "build_cost" in data:
            bc = data["build_cost"]
            cost = bc.get("production", 0) if isinstance(bc, dict) else 0
        elif "casting_cost" in data:
            cc = data["casting_cost"]
            cost = cc.get("mana", 0) if isinstance(cc, dict) else 0

        return cls(
            name=data["name"],
            figures=data.get("figures", 1),
            hp_per_figure=data.get("hp", 1),
            melee=data.get("melee", 0),
            ranged=data.get("ranged", 0),
            ranged_type=data.get("ranged_type", 0),
            defense=data.get("defense", 0),
            resist=data.get("resist", 0),
            tohit_bonus=data.get("tohit", 0),
            breath=data.get("breath", 0),
            breath_type=data.get("breath_type", 0),
            armor_piercing=abilities.get("armor_piercing", False),
            first_strike=abilities.get("first_strike", False),
            large_shield=abilities.get("large_shield", False),
            magic_immunity=abilities.get("magic_immunity", False),
            weapon_immunity=abilities.get("weapon_immunity", False),
            missile_immunity=abilities.get("missile_immunity", False),
            illusionary=abilities.get("illusionary", False),
            negate_first_strike=abilities.get("negate_first_strike", False),
            poison=abilities.get("poison", 0),
            life_steal=abilities.get("life_steal", 0),
            immolation=abilities.get("immolation", 0),
            cost=cost,
            category=data.get("category", ""),
            race=data.get("race", ""),
            realm=data.get("realm", "none"),
        )


def load_units(path: str | Path | None = None) -> dict[str, Unit]:
    """Load units from unit-data.json, return dict keyed by name."""
    if path is None:
        path = Path(__file__).parent / "unit-data.json"
    with open(path) as f:
        data = json.load(f)
    units = {}
    for entry in data["units"]:
        unit = Unit.from_json(entry)
        units[unit.name] = unit
    return units


# ---------------------------------------------------------------------------
# Layer 1: MarkovGenerator — build transition probability matrices
# ---------------------------------------------------------------------------

class MarkovGenerator:
    """
    Builds the damage probability distribution for one round of combat
    between an attacker and defender.

    Physical damage model:
    1. Attacker has `attack_strength` swords, each with `tohit` chance to hit.
    2. Defender has `defense` shields, each with 30% base chance to block.
    3. Net damage = hits - blocks (min 0).
    4. Damage applied as HP pool drain (spills across figures).

    Returns a probability distribution over net damage dealt.
    """

    @staticmethod
    def binomial_pdf(n: int, p: float) -> np.ndarray:
        """P(X = k) for k in 0..n, X ~ Binomial(n, p)."""
        if n <= 0:
            return np.array([1.0])
        k = np.arange(n + 1)
        # Use log to avoid overflow for large n
        log_comb = np.array([
            math.lgamma(n + 1) - math.lgamma(ki + 1) - math.lgamma(n - ki + 1)
            for ki in k
        ])
        log_prob = log_comb + k * np.log(max(p, 1e-15)) + (n - k) * np.log(max(1 - p, 1e-15))
        pdf = np.exp(log_prob)
        pdf /= pdf.sum()  # normalize for numerical stability
        return pdf

    @staticmethod
    def physical_damage_distribution(
        attack_strength: int,
        tohit: float,
        defense: int,
        toblock: float = 0.3,
        armor_piercing: bool = False,
        illusionary: bool = False,
        weapon_immunity: bool = False,
    ) -> np.ndarray:
        """
        Compute P(net_damage = d) for d in 0..attack_strength.

        Physical damage: roll `attack_strength` dice at `tohit` chance,
        then defender rolls `effective_defense` dice at `toblock` chance.
        Net damage = max(hits - blocks, 0).
        """
        if attack_strength <= 0:
            return np.array([1.0])

        # Modify defense for special abilities
        effective_defense = defense
        if illusionary:
            effective_defense = 0
        elif armor_piercing:
            effective_defense = defense // 2
        if weapon_immunity:
            # Weapon immunity: attacker needs magical/mithril/adamantium weapon
            # For base physical attacks, this means 0 damage
            # Simplified: treat as +10 defense shields for non-magic attacks
            effective_defense += 10

        # P(hits = h) for h in 0..attack_strength
        hit_dist = MarkovGenerator.binomial_pdf(attack_strength, tohit)

        # P(blocks = b) for b in 0..effective_defense
        if effective_defense > 0:
            block_dist = MarkovGenerator.binomial_pdf(effective_defense, toblock)
        else:
            block_dist = np.array([1.0])  # 0 defense = 0 blocks guaranteed

        # P(net_damage = d) = sum over h,b where max(h - b, 0) = d
        max_damage = attack_strength
        damage_dist = np.zeros(max_damage + 1)

        for h in range(len(hit_dist)):
            for b in range(len(block_dist)):
                d = max(h - b, 0)
                if d <= max_damage:
                    damage_dist[d] += hit_dist[h] * block_dist[b]

        # Normalize
        damage_dist /= damage_dist.sum()
        return damage_dist

    @staticmethod
    def area_damage_distribution(
        attack_strength: int,
        tohit: float,
        defense: int,
        toblock: float = 0.3,
        num_figures: int = 1,
    ) -> np.ndarray:
        """
        Area damage (Immolation, Blizzard, Fire Breath, etc.)
        Each figure is attacked independently, then damage pools.

        For a single figure: identical to physical.
        For N figures: convolve N independent per-figure damage distributions.
        """
        if num_figures <= 1 or attack_strength <= 0:
            return MarkovGenerator.physical_damage_distribution(
                attack_strength, tohit, defense, toblock
            )

        # Per-figure damage distribution
        per_fig = MarkovGenerator.physical_damage_distribution(
            attack_strength, tohit, defense, toblock
        )

        # Convolve for N figures (total damage = sum of N independent per-figure damages)
        total = per_fig.copy()
        for _ in range(num_figures - 1):
            total = np.convolve(total, per_fig)

        return total

    @staticmethod
    def figures_alive(unit: Unit, current_hp: int) -> int:
        """How many figures are alive at a given HP total.
        Damage spills: first figure dies when its HP is gone, then next, etc.
        figures_alive = ceil(current_hp / hp_per_figure)"""
        if current_hp <= 0:
            return 0
        return math.ceil(current_hp / unit.hp_per_figure)

    @staticmethod
    def build_round_transition(
        attacker: Unit,
        defender: Unit,
        attacker_hp: int | None = None,
        defender_hp: int | None = None,
        is_initiator: bool = True,
    ) -> np.ndarray:
        """
        Build the damage distribution for one round: attacker attacks defender.

        Returns P(damage = d) for d in 0..max_possible_damage.

        attacker_hp: current HP of attacker (determines surviving figures).
                     None = full health.
        defender_hp: current HP of defender (determines surviving figures for
                     area damage calculations). None = full health.

        A single-figure fantastic unit deals the same damage at 1 HP as at 30 HP.
        A 6-figure normal unit at 2 HP has only 2 figures swinging swords.
        """
        att_hp = attacker_hp if attacker_hp is not None else attacker.total_hp
        def_hp = defender_hp if defender_hp is not None else defender.total_hp

        att_figs = MarkovGenerator.figures_alive(attacker, att_hp)
        def_figs = MarkovGenerator.figures_alive(defender, def_hp)

        if att_figs <= 0:
            return np.array([1.0])  # dead attacker deals no damage

        # Melee: surviving figures * per-figure melee strength
        total_swords = att_figs * attacker.melee

        melee_dist = MarkovGenerator.physical_damage_distribution(
            attack_strength=total_swords,
            tohit=attacker.base_tohit,
            defense=defender.defense,
            toblock=0.3,
            armor_piercing=attacker.armor_piercing,
            illusionary=attacker.illusionary,
            weapon_immunity=defender.weapon_immunity,
        )

        # Ranged phase — only for initiator, figures-dependent
        if is_initiator and attacker.ranged > 0 and attacker.ranged_type > 0:
            total_ranged = att_figs * attacker.ranged
            ranged_weapon_imm = (
                defender.missile_immunity and attacker.ranged_type == 1
            )
            ranged_dist = MarkovGenerator.physical_damage_distribution(
                attack_strength=total_ranged,
                tohit=attacker.base_tohit,
                defense=defender.defense + (10 if ranged_weapon_imm else 0),
                toblock=0.3,
            )
            melee_dist = np.convolve(melee_dist, ranged_dist)

        # Breath attack — single-figure ability, NOT multiplied by attacker figs
        # But area damage hits each DEFENDER figure independently
        if attacker.breath > 0:
            breath_dist = MarkovGenerator.area_damage_distribution(
                attack_strength=attacker.breath,
                tohit=attacker.base_tohit,
                defense=defender.defense,
                toblock=0.3,
                num_figures=def_figs,
            )
            melee_dist = np.convolve(melee_dist, breath_dist)

        # Immolation — area damage, hits each defender figure independently
        if attacker.immolation > 0:
            immo_dist = MarkovGenerator.area_damage_distribution(
                attack_strength=attacker.immolation,
                tohit=attacker.base_tohit,
                defense=defender.defense,
                toblock=0.3,
                num_figures=def_figs,
            )
            melee_dist = np.convolve(melee_dist, immo_dist)

        return melee_dist

    @staticmethod
    def build_transition_matrix(
        attacker: Unit,
        defender: Unit,
        attacker_hp: int | None = None,
        is_initiator: bool = True,
    ) -> np.ndarray:
        """
        Build full Markov transition matrix for defender's HP,
        given the attacker's current HP.

        For single-figure units (fantastic creatures), this matrix is the same
        regardless of attacker HP — they hit just as hard at 1 HP as at full.

        For multi-figure units, fewer figures = fewer swords = less damage.
        The engine rebuilds this matrix as the attacker takes damage.

        attacker_hp: attacker's current HP (None = full health).
        """
        att_hp = attacker_hp if attacker_hp is not None else attacker.total_hp
        max_hp = defender.total_hp
        T = np.zeros((max_hp + 1, max_hp + 1))

        # Row 0: already dead, stays dead
        T[0][0] = 1.0

        # Compute damage distribution for this attacker HP level
        # (same for all defender HP rows — defender HP only affects
        # area damage figure count, handled inside build_round_transition)
        for hp in range(1, max_hp + 1):
            damage_dist = MarkovGenerator.build_round_transition(
                attacker, defender,
                attacker_hp=att_hp,
                defender_hp=hp,
                is_initiator=is_initiator,
            )
            for d in range(len(damage_dist)):
                new_hp = max(hp - d, 0)
                T[hp][new_hp] += damage_dist[d]

        return T


# ---------------------------------------------------------------------------
# Layer 2: BattleEngine — resolve a fight using Markov transitions
# ---------------------------------------------------------------------------

@dataclass
class BattleResult:
    """Result of a single battle between two units."""
    unit_a: str
    unit_b: str
    mode: str     # "a_attacks", "b_attacks", "alternate"
    win_a: float  # probability A wins
    win_b: float  # probability B wins
    draw: float   # probability of mutual kill
    avg_rounds: float
    avg_hp_remaining_a: float  # expected HP if A wins
    avg_hp_remaining_b: float  # expected HP if B wins


class BattleEngine:
    """
    Resolve combat between two units using Markov transitions.

    Three modes:
      "a_attacks" — A initiates: A gets ranged phase, A attacks first each round.
                    B is the defender (melee counter only, no free ranged).
      "b_attacks" — B initiates: mirror of above.
      "alternate" — Each round alternates who initiates. Round 1: A attacks,
                    Round 2: B attacks, etc. Balanced comparison for rankings.

    Combat flow per round (for the initiator):
      1. Ranged phase (initiator only — free damage, no counter)
      2. Breath on approach (initiator — area damage, no counter)
      3. First Strike melee (if applicable — resolves before normal melee)
      4. Normal melee + counter-attack (both sides, simultaneous)

    State: (hp_a, hp_b) probability distribution.
    """

    MAX_ROUNDS = 50  # safety cap

    @staticmethod
    def fight(
        unit_a: Unit,
        unit_b: Unit,
        mode: str = "alternate",
    ) -> BattleResult:
        """
        Run a full Markov battle between two units.

        mode: "a_attacks" — A is always the initiator
              "b_attacks" — B is always the initiator
              "alternate" — alternates each round (A first in round 1)
        """
        hp_a_max = unit_a.total_hp
        hp_b_max = unit_b.total_hp

        # Precompute transition matrices for each attacker HP level.
        # For single-figure units, all HP levels produce the same matrix.
        # For multi-figure units, each figure-count boundary changes the matrix.
        #
        # T_ab[hp_a] = transition matrix for B's HP when A attacks at hp_a
        # We only need one matrix per distinct figure count, so cache by figures.

        def build_matrices(attacker, defender, is_initiator):
            """Build a dict mapping attacker_hp -> transition matrix."""
            matrices = {}
            last_figs = -1
            last_matrix = None
            for hp in range(0, attacker.total_hp + 1):
                figs = MarkovGenerator.figures_alive(attacker, hp)
                if figs != last_figs:
                    last_figs = figs
                    last_matrix = MarkovGenerator.build_transition_matrix(
                        attacker, defender,
                        attacker_hp=hp,
                        is_initiator=is_initiator,
                    )
                matrices[hp] = last_matrix
            return matrices

        T_ab_init = build_matrices(unit_a, unit_b, is_initiator=True)
        T_ab_def = build_matrices(unit_a, unit_b, is_initiator=False)
        T_ba_init = build_matrices(unit_b, unit_a, is_initiator=True)
        T_ba_def = build_matrices(unit_b, unit_a, is_initiator=False)

        # State: joint probability distribution over (hp_a, hp_b)
        state = np.zeros((hp_a_max + 1, hp_b_max + 1))
        state[hp_a_max][hp_b_max] = 1.0

        # First strike flags
        a_has_fs = unit_a.first_strike and not unit_b.negate_first_strike
        b_has_fs = unit_b.first_strike and not unit_a.negate_first_strike

        for round_num in range(1, BattleEngine.MAX_ROUNDS + 1):
            alive_prob = state[1:, 1:].sum()
            if alive_prob < 1e-10:
                break

            # Determine who initiates this round
            if mode == "a_attacks":
                a_is_initiator = True
            elif mode == "b_attacks":
                a_is_initiator = False
            else:  # alternate
                a_is_initiator = (round_num % 2 == 1)

            # Select the right transition matrices for this round
            if a_is_initiator:
                T_ab = T_ab_init  # A attacks B with initiator advantage
                T_ba = T_ba_def   # B counters A as defender
            else:
                T_ab = T_ab_def   # A counters B as defender
                T_ba = T_ba_init  # B attacks A with initiator advantage

            # Phase ordering within the round:
            # The initiator's ranged/breath is baked into their T matrix.
            # First strike determines who resolves melee damage first.
            #
            # If A initiates and has first strike: A's full attack resolves,
            # then B counters (if alive).
            # If B is defending and has first strike: B strikes first in melee,
            # but A still got the free ranged phase (already in T_ab).

            if a_is_initiator:
                # A initiated: A attacks first (ranged+breath+melee in T_ab)
                # Then B counters (melee only in T_ba)
                if a_has_fs and not b_has_fs:
                    # A first strikes, then B counters
                    state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
                    state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
                elif b_has_fs and not a_has_fs:
                    # B has first strike on defense — B strikes first in melee,
                    # but A still got ranged phase. Model: B counters first,
                    # then A's melee resolves. But A's ranged already happened.
                    # Approximation: apply B's counter first, then A's full attack.
                    # This slightly overvalues defensive first strike since A's
                    # ranged should have resolved before B's first strike melee.
                    # TODO: split ranged and melee into separate phases
                    state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
                    state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
                else:
                    # No first strike advantage, or both have it
                    # Initiator still goes first (they chose to engage)
                    state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
                    state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
            else:
                # B initiated: B attacks first, A counters
                if b_has_fs and not a_has_fs:
                    state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
                    state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
                elif a_has_fs and not b_has_fs:
                    state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
                    state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
                else:
                    state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
                    state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)

        # Extract results
        win_a = state[1:, 0].sum()
        win_b = state[0, 1:].sum()
        draw = state[0, 0]

        avg_hp_a = 0.0
        if win_a > 0:
            for i in range(1, hp_a_max + 1):
                avg_hp_a += i * state[i, 0]
            avg_hp_a /= win_a

        avg_hp_b = 0.0
        if win_b > 0:
            for j in range(1, hp_b_max + 1):
                avg_hp_b += j * state[0, j]
            avg_hp_b /= win_b

        return BattleResult(
            unit_a=unit_a.name,
            unit_b=unit_b.name,
            mode=mode,
            win_a=round(win_a, 6),
            win_b=round(win_b, 6),
            draw=round(draw, 6),
            avg_rounds=round_num,
            avg_hp_remaining_a=round(avg_hp_a, 2),
            avg_hp_remaining_b=round(avg_hp_b, 2),
        )

    @staticmethod
    def _apply_attack_ab(state, T_ab_by_hp, hp_a_max, hp_b_max):
        """A attacks B: transition B's HP using A's HP-dependent matrix."""
        new_state = np.zeros_like(state)
        # Row 0 (A dead): no attack, state unchanged
        new_state[0, :] += state[0, :]
        # Rows 1+ (A alive): use the transition matrix for this HP level
        for i in range(1, hp_a_max + 1):
            b_dist = state[i, :]
            T = T_ab_by_hp[i]
            new_b_dist = b_dist @ T
            new_state[i, :] += new_b_dist
        return new_state

    @staticmethod
    def _apply_attack_ba(state, T_ba_by_hp, hp_a_max, hp_b_max):
        """B attacks A: transition A's HP using B's HP-dependent matrix."""
        new_state = np.zeros_like(state)
        # Col 0 (B dead): no attack, state unchanged
        new_state[:, 0] += state[:, 0]
        # Cols 1+ (B alive): use the transition matrix for this HP level
        for j in range(1, hp_b_max + 1):
            a_dist = state[:, j]
            T = T_ba_by_hp[j]
            new_a_dist = a_dist @ T
            new_state[:, j] += new_a_dist
        return new_state


# ---------------------------------------------------------------------------
# Layer 3: Arena — run matchups across groups of units
# ---------------------------------------------------------------------------

@dataclass
class ArenaResult:
    """Summary of an arena matchup."""
    unit: str
    wins: int
    losses: int
    draws: int
    avg_win_rate: float
    matchups: list  # list of BattleResult


class Arena:
    """Run round-robin or targeted matchups across a pool of units."""

    @staticmethod
    def round_robin(
        units: list[Unit],
        mode: str = "alternate",
        verbose: bool = True,
    ) -> list[ArenaResult]:
        """
        Run every unit against every other unit.

        mode: "alternate" — balanced (default for rankings)
              "a_attacks" — first unit always initiates
              "both"      — run both a_attacks and b_attacks, average results
        """
        n = len(units)
        results = {u.name: ArenaResult(
            unit=u.name, wins=0, losses=0, draws=0,
            avg_win_rate=0.0, matchups=[]
        ) for u in units}

        total = n * (n - 1) // 2
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                count += 1
                if verbose:
                    print(f"  [{count}/{total}] {units[i].name} vs {units[j].name}", end="")

                if mode == "both":
                    # Run both directions, average the win rates
                    r1 = BattleEngine.fight(units[i], units[j], mode="a_attacks")
                    r2 = BattleEngine.fight(units[i], units[j], mode="b_attacks")
                    avg_win_a = (r1.win_a + r2.win_a) / 2
                    avg_win_b = (r1.win_b + r2.win_b) / 2
                    battle = BattleResult(
                        unit_a=units[i].name, unit_b=units[j].name,
                        mode="both",
                        win_a=round(avg_win_a, 6), win_b=round(avg_win_b, 6),
                        draw=round(1 - avg_win_a - avg_win_b, 6),
                        avg_rounds=round((r1.avg_rounds + r2.avg_rounds) / 2),
                        avg_hp_remaining_a=round((r1.avg_hp_remaining_a + r2.avg_hp_remaining_a) / 2, 2),
                        avg_hp_remaining_b=round((r1.avg_hp_remaining_b + r2.avg_hp_remaining_b) / 2, 2),
                    )
                else:
                    battle = BattleEngine.fight(units[i], units[j], mode=mode)

                if verbose:
                    winner = units[i].name if battle.win_a > battle.win_b else units[j].name
                    pct = max(battle.win_a, battle.win_b) * 100
                    print(f" -> {winner} ({pct:.1f}%)")

                results[units[i].name].matchups.append(battle)
                results[units[j].name].matchups.append(battle)

                if battle.win_a > battle.win_b:
                    results[units[i].name].wins += 1
                    results[units[j].name].losses += 1
                elif battle.win_b > battle.win_a:
                    results[units[j].name].wins += 1
                    results[units[i].name].losses += 1
                else:
                    results[units[i].name].draws += 1
                    results[units[j].name].draws += 1

        # Compute win rates
        for r in results.values():
            total_fights = r.wins + r.losses + r.draws
            r.avg_win_rate = r.wins / total_fights if total_fights > 0 else 0.0

        return sorted(results.values(), key=lambda r: r.avg_win_rate, reverse=True)

    @staticmethod
    def matchup(unit_a: Unit, unit_b: Unit, mode: str = "alternate") -> BattleResult:
        """Single matchup between two units."""
        return BattleEngine.fight(unit_a, unit_b, mode=mode)

    @staticmethod
    def full_matchup(unit_a: Unit, unit_b: Unit) -> dict:
        """Run all modes and return the asymmetry analysis."""
        r_alt = BattleEngine.fight(unit_a, unit_b, mode="alternate")
        r_a_atk = BattleEngine.fight(unit_a, unit_b, mode="a_attacks")
        r_b_atk = BattleEngine.fight(unit_a, unit_b, mode="b_attacks")
        return {
            "alternate": r_alt,
            "a_attacks": r_a_atk,
            "b_attacks": r_b_atk,
            "attack_advantage_a": round(r_a_atk.win_a - r_b_atk.win_a, 4),
            "attack_advantage_b": round(r_b_atk.win_b - r_a_atk.win_b, 4),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import sys

    units = load_units()
    print(f"Loaded {len(units)} units\n")

    if len(sys.argv) >= 3 and sys.argv[1] != "test":
        # Run a specific matchup with full asymmetry analysis
        name_a, name_b = sys.argv[1], sys.argv[2]
        if name_a not in units:
            print(f"Unknown unit: {name_a}")
            print(f"Available: {', '.join(sorted(units.keys()))}")
            return
        if name_b not in units:
            print(f"Unknown unit: {name_b}")
            return

        results = Arena.full_matchup(units[name_a], units[name_b])

        print(f"{name_a} vs {name_b}")
        print(f"{'='*55}")
        for mode_name, label in [
            ("a_attacks", f"{name_a} attacks"),
            ("b_attacks", f"{name_b} attacks"),
            ("alternate", "Alternating"),
        ]:
            r = results[mode_name]
            print(f"\n  {label}:")
            print(f"    {name_a} wins: {r.win_a * 100:>6.1f}%  (avg {r.avg_hp_remaining_a:.1f} HP left)")
            print(f"    {name_b} wins: {r.win_b * 100:>6.1f}%  (avg {r.avg_hp_remaining_b:.1f} HP left)")
            print(f"    Draw:         {r.draw * 100:>6.1f}%")
            print(f"    Rounds: {r.avg_rounds}")

        adv_a = results["attack_advantage_a"]
        adv_b = results["attack_advantage_b"]
        print(f"\n  Attack advantage:")
        print(f"    {name_a}: {adv_a*100:+.1f}% when attacking vs defending")
        print(f"    {name_b}: {adv_b*100:+.1f}% when attacking vs defending")

    elif len(sys.argv) == 2 and sys.argv[1] == "test":
        # Run a small test arena
        test_names = [
            "Great Drake", "Sky Drake", "Paladins",
            "Barbarian Swordsmen", "War Bears", "Wraiths",
        ]
        test_units = []
        for name in test_names:
            # Try exact match first, then fuzzy
            if name in units:
                test_units.append(units[name])
            else:
                matches = [u for n, u in units.items() if name.lower() in n.lower()]
                if matches:
                    test_units.append(matches[0])
                else:
                    print(f"  Skipping unknown unit: {name}")

        print(f"Test arena: {len(test_units)} units")
        print(f"{'='*50}\n")

        rankings = Arena.round_robin(test_units)

        print(f"\n{'='*50}")
        print(f"{'Unit':<25} {'W':>3} {'L':>3} {'D':>3} {'Win%':>7}")
        print(f"{'-'*25} {'---':>3} {'---':>3} {'---':>3} {'------':>7}")
        for r in rankings:
            print(f"{r.unit:<25} {r.wins:>3} {r.losses:>3} {r.draws:>3} {r.avg_win_rate*100:>6.1f}%")

    else:
        print("Usage:")
        print(f"  python {sys.argv[0]} test                  — run test arena")
        print(f"  python {sys.argv[0]} 'Unit A' 'Unit B'     — single matchup")


if __name__ == "__main__":
    main()
