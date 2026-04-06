"""
Master of Magic Tech Tree Scraper
Builds cross-reference tables for:
  1. Building prerequisites and unlocks
  2. Race -> building availability
  3. Building -> unit unlocks
"""

import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_URL = "https://masterofmagic.fandom.com/api.php"
HEADERS = {"User-Agent": "MoMAnalysis/1.0 (research project)"}
DATA_DIR = Path(__file__).parent.parent / "data"
DELAY = 0.5

RACES = [
    "Barbarians", "Beastmen", "Dark Elves", "Draconians", "Dwarves",
    "Gnolls", "Halflings", "High Elves", "High Men", "Klackons",
    "Lizardmen", "Nomads", "Orcs", "Trolls",
]


def api_request(params: dict) -> dict:
    params["format"] = "json"
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def get_page_wikitext(title: str) -> str | None:
    data = api_request({"action": "parse", "page": title, "prop": "wikitext"})
    if "parse" in data:
        return data["parse"]["wikitext"]["*"]
    return None


def parse_building_prerequisites(town_building_text: str) -> dict:
    """
    Parse the master building table from the Town Building page.
    Returns {building_name: {cost, upkeep, requires: [building names], terrain: [terrain reqs]}}
    """
    buildings = {}
    terrain_keywords = {"Forest", "Shore", "Hill", "Mountain", "Volcano", "Ocean"}

    # Split on any valign=top row separator (with or without quotes)
    rows = re.split(r'\|-\s*valign=(?:top|"top")', town_building_text)
    for row in rows[1:]:
        # Split row into cells by newline-pipe
        cells = [c.strip() for c in row.split("\n|") if c.strip()]
        if len(cells) < 4:
            continue

        # Cell 0: icon image, Cell 1: [[Building Name]], Cell 2: cost, Cell 3: upkeep
        # Cell 4: prerequisites (building icons), Cell 5: effects/benefits

        # Extract building name from cell 1 (or cell 0 if merged)
        building_name = None
        for cell in cells[:2]:
            name_match = re.findall(r"\[\[([A-Za-z][^\]|]+?)\]\]", cell)
            for nm in name_match:
                if not nm.startswith("File:"):
                    building_name = nm
                    break
            if building_name:
                break
        if not building_name:
            continue

        # Cost: find the first cell that's just a number
        cost = 0
        for cell in cells[1:4]:
            cell_clean = cell.strip().strip("|").strip()
            if cell_clean.isdigit() and int(cell_clean) > 0:
                cost = int(cell_clean)
                break

        # Upkeep: first {{Gold|N}} in the row
        gold_match = re.search(r"\{\{Gold\|(\d+)\}\}", row)
        upkeep = int(gold_match.group(1)) if gold_match else 0

        # Prerequisites: find the cell with TownBuilding icons AFTER cost/upkeep
        # The prerequisite cell is typically cell index 4 (5th cell)
        prereqs = []
        terrain = []
        prereq_cell = cells[4] if len(cells) > 4 else ""
        prereq_links = re.findall(r"link=([A-Z][^\]|\"]+?)(?:\]\]|\")", prereq_cell)
        for p in prereq_links:
            p = p.strip()
            if p == building_name:
                continue
            if p in terrain_keywords:
                terrain.append(p)
            elif "Unit" not in p and "Icon" not in p and "Flowchart" not in p:
                prereqs.append(p)

        # Deduplicate
        seen = set()
        unique_prereqs = []
        for p in prereqs:
            if p not in seen:
                seen.add(p)
                unique_prereqs.append(p)

        # Effects (last cell)
        effects = ""
        if len(cells) > 5:
            raw = cells[5]
            raw = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", raw)
            raw = re.sub(r"\{\{[^}]*\}\}", "", raw)
            raw = re.sub(r"<br\s*/?>", "; ", raw)
            effects = raw.strip()

        buildings[building_name] = {
            "cost": cost,
            "upkeep": upkeep,
            "requires": unique_prereqs,
            "terrain": list(set(terrain)),
            "effects": effects if effects else None,
        }

    return buildings


def compute_unlocks(buildings: dict) -> dict:
    """Compute what each building unlocks (inverse of prerequisites)."""
    unlocks = {name: [] for name in buildings}
    for name, data in buildings.items():
        for prereq in data["requires"]:
            if prereq in unlocks:
                unlocks[prereq].append(name)
    return unlocks


def parse_race_buildings(race: str) -> list[str]:
    """
    Parse a race's building chart template to find which buildings are available.
    Buildings appear as image links with class="avail".
    """
    template_name = f"Template:BuildingChart {race}"
    print(f"  Fetching {template_name}...", end="")
    time.sleep(DELAY)
    wikitext = get_page_wikitext(template_name)
    if not wikitext:
        print(" SKIP (not found)")
        return []

    # Extract building names from link= attributes
    # Available buildings have class="avail" or class="availocean"
    # Unavailable have class="unavail"
    # Parse by finding all building links and checking their cell class

    available = set()

    # Split into table cells
    cells = re.split(r"\|\s*(?:class=)", wikitext)
    for cell in cells:
        is_avail = cell.startswith('"avail')
        is_unavail = cell.startswith('"unavail')

        if is_avail and not is_unavail:
            # Extract building names from this cell
            links = re.findall(r"[Ll]ink=([A-Z][^\]|\"]+?)(?:\]\]|\")", cell)
            for link in links:
                link = link.strip()
                if "Flowchart" not in link and "Unit" not in link:
                    available.add(link)

    print(f" {len(available)} buildings")
    return sorted(available)


def parse_unit_requirements() -> dict:
    """
    For each race, find which units they can build and what buildings are required.
    Uses the unit data we already scraped.
    """
    units_path = DATA_DIR / "units.json"
    with open(units_path) as f:
        units_data = json.load(f)

    # Build race -> [units] mapping from scraped data
    race_units = {}
    for unit in units_data:
        if unit.get("infobox_type") != "Normal Unit":
            continue
        race = unit.get("race", "Unknown")
        if race not in race_units:
            race_units[race] = []
        race_units[race].append({
            "name": unit["name"],
            "requires": unit.get("requires", []),
            "build_cost": unit.get("build_cost", {}),
        })

    return race_units


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Load hand-verified building prerequisites
    #    (Table parsing was too fragile; this data is static for the 1994 game)
    print("=== Loading building prerequisites ===")
    with open(DATA_DIR / "building-prereqs.json") as f:
        buildings = json.load(f)
        # Remove metadata key
        buildings.pop("_note", None)
    print(f"  Loaded {len(buildings)} buildings")

    # 2. Compute unlocks (inverse of prerequisites)
    unlocks = compute_unlocks(buildings)
    for name in buildings:
        buildings[name]["unlocks"] = unlocks.get(name, [])

    # 3. Parse race -> building availability
    print("\n=== Parsing race building availability ===")
    race_buildings = {}
    for race in RACES:
        race_buildings[race] = parse_race_buildings(race)

    # 4. Build building -> race availability (inverse)
    building_races = {}
    for race, blds in race_buildings.items():
        for b in blds:
            if b not in building_races:
                building_races[b] = []
            building_races[b].append(race)

    # Add race availability to building data
    for name in buildings:
        buildings[name]["available_to"] = sorted(building_races.get(name, []))

    # 5. Get unit requirements
    print("\n=== Parsing unit requirements ===")
    race_units = parse_unit_requirements()
    for race, units in sorted(race_units.items()):
        print(f"  {race}: {len(units)} units")

    # 6. Write outputs
    tech_tree = {
        "buildings": buildings,
        "race_buildings": race_buildings,
        "race_units": race_units,
    }

    output_path = DATA_DIR / "tech-tree.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tech_tree, f, indent=2, ensure_ascii=False)
    print(f"\nWrote tech tree to {output_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  Buildings: {len(buildings)}")
    print(f"  Races: {len(race_buildings)}")

    # Show a sample
    print("\n=== SAMPLE: Smithy ===")
    if "Smithy" in buildings:
        s = buildings["Smithy"]
        print(f"  Cost: {s['cost']}, Upkeep: {s['upkeep']}")
        print(f"  Requires: {s['requires']}")
        print(f"  Unlocks: {s['unlocks']}")
        print(f"  Available to: {s['available_to']}")

    print("\n=== SAMPLE: Armorers' Guild ===")
    ag = buildings.get("Armorers' Guild", {})
    if ag:
        print(f"  Cost: {ag['cost']}, Upkeep: {ag['upkeep']}")
        print(f"  Requires: {ag['requires']}")
        print(f"  Unlocks: {ag['unlocks']}")
        print(f"  Available to: {ag['available_to']}")


if __name__ == "__main__":
    main()
