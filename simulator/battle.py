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
    def build_round_transition(attacker: Unit, defender: Unit) -> np.ndarray:
        """
        Build the damage distribution for one round: attacker attacks defender.

        Returns P(damage = d) for d in 0..max_possible_damage.
        Combines melee/ranged + breath + special attacks.

        For now: physical melee/ranged only. Breath and specials to be added.
        """
        # Determine active figures for attacker based on some assumed HP
        # For the transition matrix, we compute per-figure-count distributions
        # and the engine will select the right one based on current HP state.

        # For simplicity in v1: compute at full strength (all figures alive)
        # Each figure attacks independently with the unit's attack strength
        att_figs = attacker.figures
        att_strength = attacker.melee  # per figure

        # Total attack = figures * per-figure strength (each figure swings independently)
        # But in MoM, it's not that simple — each figure makes `attack_strength` rolls
        # So total swords = figures_alive * melee_strength
        total_swords = att_figs * att_strength

        # Melee physical damage
        melee_dist = MarkovGenerator.physical_damage_distribution(
            attack_strength=total_swords,
            tohit=attacker.base_tohit,
            defense=defender.defense,
            toblock=0.3,
            armor_piercing=attacker.armor_piercing,
            illusionary=attacker.illusionary,
            weapon_immunity=defender.weapon_immunity,
        )

        # Breath attack (area damage — hits each defender figure independently)
        if attacker.breath > 0:
            breath_dist = MarkovGenerator.area_damage_distribution(
                attack_strength=attacker.breath,
                tohit=attacker.base_tohit,
                defense=defender.defense,
                toblock=0.3,
                num_figures=defender.figures,
            )
            # Combine: total damage = melee damage + breath damage (independent)
            melee_dist = np.convolve(melee_dist, breath_dist)

        return melee_dist

    @staticmethod
    def build_transition_matrix(attacker: Unit, defender: Unit) -> np.ndarray:
        """
        Build full Markov transition matrix for defender's HP.

        Matrix T where T[i][j] = P(defender goes from i HP to j HP)
        after one attack by attacker at current figure count.

        Rows = current HP (0 to total_hp)
        Cols = resulting HP (0 to total_hp)

        The attacker's damage depends on how many of ITS figures are alive,
        but for v1 we assume full-strength attacker. The engine handles
        attrition by rebuilding per round.
        """
        max_hp = defender.total_hp
        T = np.zeros((max_hp + 1, max_hp + 1))

        # Row 0: already dead, stays dead
        T[0][0] = 1.0

        # For each possible current HP, compute damage distribution
        # and fill transition probabilities
        for hp in range(1, max_hp + 1):
            # Attacker's figures alive depends on attacker's current HP
            # For v1, compute damage at full attacker strength
            damage_dist = MarkovGenerator.build_round_transition(attacker, defender)

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
    win_a: float  # probability A wins
    win_b: float  # probability B wins
    draw: float   # probability of mutual kill
    avg_rounds: float
    avg_hp_remaining_a: float  # expected HP if A wins
    avg_hp_remaining_b: float  # expected HP if B wins


class BattleEngine:
    """
    Resolve combat between two units using alternating Markov transitions.

    Each round:
    1. A attacks B (B's HP transitions)
    2. B attacks A (A's HP transitions)
    Repeat until one or both are dead.

    State: (hp_a, hp_b) probability distribution.
    """

    MAX_ROUNDS = 50  # safety cap

    @staticmethod
    def fight(unit_a: Unit, unit_b: Unit) -> BattleResult:
        """
        Run a full Markov battle between two units.

        Handles first strike: if A has first strike (and B doesn't negate it),
        A attacks first each round before B can counter.
        """
        hp_a_max = unit_a.total_hp
        hp_b_max = unit_b.total_hp

        # Build transition matrices
        # T_ab[i][j] = P(B goes from i HP to j HP) when A attacks B
        T_ab = MarkovGenerator.build_transition_matrix(unit_a, unit_b)
        # T_ba[i][j] = P(A goes from i HP to j HP) when B attacks A
        T_ba = MarkovGenerator.build_transition_matrix(unit_b, unit_a)

        # State: joint probability distribution over (hp_a, hp_b)
        # Start at full HP with probability 1.0
        state = np.zeros((hp_a_max + 1, hp_b_max + 1))
        state[hp_a_max][hp_b_max] = 1.0

        # Determine attack order
        a_has_fs = unit_a.first_strike and not unit_b.negate_first_strike
        b_has_fs = unit_b.first_strike and not unit_a.negate_first_strike

        for round_num in range(1, BattleEngine.MAX_ROUNDS + 1):
            # Check if combat is resolved
            alive_prob = state[1:, 1:].sum()
            if alive_prob < 1e-10:
                break

            if a_has_fs and not b_has_fs:
                # A attacks first, then B (if alive) counters
                state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
                state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
            elif b_has_fs and not a_has_fs:
                # B attacks first, then A counters
                state = BattleEngine._apply_attack_ba(state, T_ba, hp_a_max, hp_b_max)
                state = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
            else:
                # Simultaneous: both attack, resolve damage together
                state = BattleEngine._apply_simultaneous(
                    state, T_ab, T_ba, hp_a_max, hp_b_max
                )

        # Extract results
        win_a = state[1:, 0].sum()   # A alive, B dead
        win_b = state[0, 1:].sum()   # B alive, A dead
        draw = state[0, 0]            # both dead

        # Expected remaining HP when winning
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
            win_a=round(win_a, 6),
            win_b=round(win_b, 6),
            draw=round(draw, 6),
            avg_rounds=round_num,
            avg_hp_remaining_a=round(avg_hp_a, 2),
            avg_hp_remaining_b=round(avg_hp_b, 2),
        )

    @staticmethod
    def _apply_attack_ab(state, T_ab, hp_a_max, hp_b_max):
        """A attacks B: transition B's HP for each row where A is alive."""
        new_state = np.zeros_like(state)
        # Row 0 (A dead): no attack, state unchanged
        new_state[0, :] += state[0, :]
        # Rows 1+ (A alive): B's HP transitions according to T_ab
        for i in range(1, hp_a_max + 1):
            # state[i, :] is the distribution over B's HP when A has i HP
            # Multiply by transition matrix to get new B HP distribution
            b_dist = state[i, :]
            new_b_dist = b_dist @ T_ab  # matrix multiply: row vector × matrix
            new_state[i, :] += new_b_dist
        return new_state

    @staticmethod
    def _apply_attack_ba(state, T_ba, hp_a_max, hp_b_max):
        """B attacks A: transition A's HP for each column where B is alive."""
        new_state = np.zeros_like(state)
        # Col 0 (B dead): no attack, state unchanged
        new_state[:, 0] += state[:, 0]
        # Cols 1+ (B alive): A's HP transitions according to T_ba
        for j in range(1, hp_b_max + 1):
            a_dist = state[:, j]
            new_a_dist = a_dist @ T_ba
            new_state[:, j] += new_a_dist
        return new_state

    @staticmethod
    def _apply_simultaneous(state, T_ab, T_ba, hp_a_max, hp_b_max):
        """Both attack simultaneously. Apply both transitions independently."""
        # First A attacks B
        intermediate = BattleEngine._apply_attack_ab(state, T_ab, hp_a_max, hp_b_max)
        # Then B attacks A (using pre-damage state for B's figure count)
        # For simultaneous, both use the state BEFORE this round's damage
        # Approximation: apply sequentially (A then B). True simultaneous
        # would require a joint 4D transition which is overkill for v1.
        result = BattleEngine._apply_attack_ba(intermediate, T_ba, hp_a_max, hp_b_max)
        return result


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
    def round_robin(units: list[Unit], verbose: bool = True) -> list[ArenaResult]:
        """
        Run every unit against every other unit.
        Returns per-unit summary stats.
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

                battle = BattleEngine.fight(units[i], units[j])

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
    def matchup(unit_a: Unit, unit_b: Unit) -> BattleResult:
        """Single matchup between two units."""
        return BattleEngine.fight(unit_a, unit_b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import sys

    units = load_units()
    print(f"Loaded {len(units)} units\n")

    if len(sys.argv) >= 3:
        # Run a specific matchup
        name_a, name_b = sys.argv[1], sys.argv[2]
        if name_a not in units:
            print(f"Unknown unit: {name_a}")
            print(f"Available: {', '.join(sorted(units.keys()))}")
            return
        if name_b not in units:
            print(f"Unknown unit: {name_b}")
            return

        result = Arena.matchup(units[name_a], units[name_b])
        print(f"{result.unit_a} vs {result.unit_b}")
        print(f"  A wins: {result.win_a * 100:.1f}%")
        print(f"  B wins: {result.win_b * 100:.1f}%")
        print(f"  Draw:   {result.draw * 100:.1f}%")
        print(f"  Rounds: {result.avg_rounds}")
        print(f"  Avg HP remaining (A wins): {result.avg_hp_remaining_a:.1f}")
        print(f"  Avg HP remaining (B wins): {result.avg_hp_remaining_b:.1f}")

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
