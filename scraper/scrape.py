"""
Master of Magic Wiki Scraper
Pulls unit, building, and spell data from masterofmagic.fandom.com
via the MediaWiki API and parses infobox templates into structured JSON.
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
DELAY = 0.5  # be polite to the wiki


def api_request(params: dict) -> dict:
    params["format"] = "json"
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def get_category_members(category: str) -> list[str]:
    """Get all page titles in a category, handling pagination."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "100",
        "cmtype": "page",
    }
    while True:
        data = api_request(params)
        for member in data["query"]["categorymembers"]:
            titles.append(member["title"])
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
        time.sleep(DELAY)
    return titles


def get_page_wikitext(title: str) -> str | None:
    """Fetch the raw wikitext for a page."""
    data = api_request({"action": "parse", "page": title, "prop": "wikitext"})
    if "parse" in data:
        return data["parse"]["wikitext"]["*"]
    return None


def extract_infobox(wikitext: str) -> tuple[str, str] | None:
    """Extract the first {{Infobox ...}} block from wikitext.
    Returns (infobox_type, raw_infobox_text) or None."""
    start = wikitext.find("{{Infobox")
    if start == -1:
        return None
    depth = 0
    i = start
    while i < len(wikitext) - 1:
        if wikitext[i : i + 2] == "{{":
            depth += 1
            i += 2
        elif wikitext[i : i + 2] == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                break
        else:
            i += 1
    raw = wikitext[start:i]
    # Extract type: {{Infobox Normal Unit -> "Normal Unit"
    type_match = re.match(r"\{\{Infobox\s+(.+?)[\n|]", raw)
    infobox_type = type_match.group(1).strip() if type_match else "Unknown"
    return infobox_type, raw


def parse_template_value(value: str) -> str | int | float | dict:
    """Parse a single template value like {{Mana|900}} or {{Melee|Normal|3}} *"""
    value = value.strip()
    has_asterisk = value.endswith("*")
    if has_asterisk:
        value = value[:-1].strip()

    # Match {{Template|arg1|arg2|...}} — case insensitive for {{resist|12}} etc.
    templates = re.findall(r"\{\{(\w+)\|([^}]*)\}\}", value, re.IGNORECASE)
    if not templates:
        # Plain text — strip wiki links
        clean = re.sub(r"\[\[(?:File:[^\]]*\|)?(?:link=[^\]]*\]\])", "", value)
        clean = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", clean)
        clean = clean.strip()
        if has_asterisk:
            return {"value": clean, "modified": True}
        return clean

    results = {}
    for tmpl_name, tmpl_args in templates:
        args = [a.strip() for a in tmpl_args.split("|")]

        # Normalize template name to title case for matching
        tmpl_key = tmpl_name.capitalize()

        if tmpl_key in ("Production", "Gold", "Mana", "Research", "Food"):
            # Resource value: {{Gold|1}} or {{Mana|5-25}}
            val_str = args[0]
            if "-" in val_str:
                parts = val_str.split("-")
                results[tmpl_key.lower()] = {"min": int(parts[0]), "max": int(parts[1])}
            else:
                try:
                    results[tmpl_key.lower()] = int(val_str)
                except ValueError:
                    results[tmpl_key.lower()] = val_str

        elif tmpl_key in ("Melee", "Ranged", "Breath"):
            # Attack: {{Melee|Normal|3}} or {{Ranged|Bow|3}}
            damage_type = args[0] if len(args) > 1 else "Normal"
            strength = args[-1]
            try:
                strength = int(strength)
            except ValueError:
                pass
            results["type"] = tmpl_name.lower()
            results["damage_type"] = damage_type
            results["strength"] = strength

        elif tmpl_key in ("Defense", "Resist", "Hits"):
            try:
                results["value"] = int(args[0])
            except ValueError:
                results["value"] = args[0]

        elif tmpl_name.upper() in ("MFU", "SFU"):
            results["figures"] = int(args[0])
            results["multi"] = tmpl_name.upper() == "MFU"

        elif tmpl_key == "Movement":
            results["type"] = args[0] if args else "Ground"
            try:
                results["speed"] = int(args[1]) if len(args) > 1 else 1
            except ValueError:
                results["speed"] = args[1] if len(args) > 1 else 1

        else:
            # Generic template — store as-is
            results[tmpl_name.lower()] = args[0] if len(args) == 1 else args

    if has_asterisk:
        results["modified"] = True

    # If we only got a single simple value, unwrap it
    if len(results) == 1 and "value" in results:
        return results["value"]

    return results if results else value


def parse_abilities(raw: str) -> list[dict]:
    """Parse the abilities field into a list of ability dicts."""
    abilities = []
    # Split on newlines — each ability is typically on its own line
    lines = raw.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Strip image/file links
        line = re.sub(r"\[\[File:[^\]]*\]\]", "", line).strip()

        # Extract ability name from wiki links: [[Ability Name]]
        ability_match = re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]*)?\]\]", line)

        # Look for numeric values (e.g., "Fire Breath 30", "Poison Touch 1")
        num_match = re.search(r"(\d+)\s*$", line)

        # Look for percentage modifiers (e.g., "+30% To Hit")
        pct_match = re.search(r"([+-]?\d+)%", line)

        # Look for mana values in Caster ability
        mana_match = re.search(r"\{\{Mana\|(\d+)\}\}", line)

        ability = {}
        if ability_match:
            # Filter out generic link names
            names = [n for n in ability_match if n not in ("To Hit", "Super Ability")]
            if not names:
                names = ability_match
            ability["name"] = names[-1] if names else line

            # Check for "Super" prefix
            if "Super" in line or "Super Ability" in str(ability_match):
                ability["super"] = True
        else:
            # Plain text ability
            ability["name"] = re.sub(r"\{\{[^}]*\}\}", "", line).strip()

        if num_match and ability.get("name") != ability_match[-1] if ability_match else True:
            ability["strength"] = int(num_match.group(1))
        elif num_match:
            # Number at end of line after ability name
            ability["strength"] = int(num_match.group(1))

        if pct_match:
            ability["modifier"] = f"{pct_match.group(1)}%"

        if mana_match:
            ability["mana"] = int(mana_match.group(1))

        if ability.get("name"):
            abilities.append(ability)

    return abilities


def parse_infobox(infobox_type: str, raw: str) -> dict:
    """Parse an infobox into a structured dict."""
    result = {"infobox_type": infobox_type}

    # Extract field=value pairs
    # Fields start with |fieldname and end at the next |fieldname or }}
    # Allow optional whitespace before the pipe (Hero infoboxes use "| field")
    field_pattern = re.compile(r"\n\s*\|\s*(\w+)\s*=\s*(.*?)(?=\n\s*\||\n\s*\}\})", re.DOTALL)
    for match in field_pattern.finditer(raw):
        field_name = match.group(1).strip()
        field_value = match.group(2).strip()

        if not field_value or field_value.startswith("[[File:") and field_name == "image":
            if field_name == "image":
                continue  # skip images
            continue

        if field_name == "abilities":
            result["abilities"] = parse_abilities(field_value)
        elif field_name == "extra_spells":
            result["extra_spells"] = [
                s.strip().strip("*").strip().strip("[").strip("]")
                for s in field_value.split("*")
                if s.strip() and not s.strip().startswith("<")
            ]
        elif field_name == "upkeep_cost":
            # Can have multiple: {{Gold|1}}<br>{{Food|1}} or {{Mana|30}} per turn
            parts = re.split(r"<br\s*/?>", field_value)
            upkeep = {}
            for part in parts:
                part = part.replace("per turn", "").strip()
                parsed = parse_template_value(part)
                if isinstance(parsed, dict):
                    upkeep.update(parsed)
                elif isinstance(parsed, int):
                    upkeep["unknown"] = parsed
            result["upkeep"] = upkeep
        elif field_name == "req":
            # Prerequisites: [[Barracks]],<br>[[Smithy]]
            reqs = re.findall(r"\[\[([^\]|]+)\]\]", field_value)
            result["requires"] = reqs
        elif field_name in ("melee", "ranged"):
            parsed = parse_template_value(field_value)
            if isinstance(parsed, dict) and "strength" in parsed:
                result[field_name] = parsed
            elif field_value.strip():
                result[field_name] = parse_template_value(field_value)
        elif field_name == "num_figures":
            result["figures"] = parse_template_value(field_value)
        elif field_name == "moves":
            result["movement"] = parse_template_value(field_value)
        elif field_name in ("defense", "resist", "hits"):
            result[field_name] = parse_template_value(field_value)
        elif field_name in ("build_cost", "casting_cost", "construction_cost", "research_cost", "hiring_cost"):
            result[field_name] = parse_template_value(field_value)
        elif field_name == "race":
            race_match = re.findall(r"\[\[([^\]|]+)\]\]", field_value)
            result["race"] = race_match[0] if race_match else field_value
        elif field_name == "color":
            color_match = re.search(r"\{\{(\w+)", field_value)
            result["realm"] = color_match.group(1) if color_match else field_value
        elif field_name == "rarity":
            rarity_match = re.search(r"\[\[.*?\|(.*?)\]\]", field_value)
            if rarity_match:
                result["rarity"] = rarity_match.group(1)
            else:
                result["rarity"] = re.sub(r"\[\[|\]\]", "", field_value).strip()
        elif field_name == "type":
            result["spell_type"] = re.sub(r"\[\[|\]\]", "", field_value).strip()
        elif field_name == "effects":
            result["effects"] = field_value
        elif field_name in ("hero_name", "hero_class", "hero_type"):
            clean = re.sub(r"\[\[|\]\]", "", field_value).strip()
            result[field_name] = clean
        elif field_name.startswith("item_slot"):
            slots = result.get("item_slots", [])
            slot_match = re.search(r"ItemSlot_(\w+)", field_value)
            if slot_match:
                slots.append(slot_match.group(1))
            result["item_slots"] = slots
        elif field_name == "random_abilities":
            result["random_abilities"] = field_value
        elif field_name == "sells_for":
            result["sells_for"] = parse_template_value(field_value)
        elif field_name in ("building_unlock", "unit_unlock"):
            # Complex template — extract linked page names
            unlocks = re.findall(r"link=([^\]|]+)", field_value)
            if unlocks:
                result[field_name] = unlocks

    return result


def scrape_category(category: str, label: str) -> list[dict]:
    """Scrape all pages in a category and parse their infoboxes."""
    print(f"\n--- Scraping {label} (Category:{category}) ---")
    titles = get_category_members(category)
    print(f"  Found {len(titles)} pages")

    results = []
    for i, title in enumerate(titles):
        print(f"  [{i+1}/{len(titles)}] {title}", end="")
        time.sleep(DELAY)
        wikitext = get_page_wikitext(title)
        if not wikitext:
            print(" - SKIP (no wikitext)")
            continue

        infobox = extract_infobox(wikitext)
        if not infobox:
            print(" - SKIP (no infobox)")
            continue

        infobox_type, raw = infobox
        parsed = parse_infobox(infobox_type, raw)
        parsed["name"] = title
        parsed["wiki_url"] = f"https://masterofmagic.fandom.com/wiki/{title.replace(' ', '_')}"
        results.append(parsed)
        print(f" - OK ({infobox_type})")

    return results


def scrape_spells() -> list[dict]:
    """Scrape spells — they're spread across realm categories."""
    all_spells = []
    # Spell categories map to realm-specific lists
    spell_categories = [
        "Combat Instants",
        "Combat Enchantments",
        "Unit Enchantments",
        "Town Enchantments",
        "Global Enchantments",
        "Instant Spells",
    ]

    seen = set()
    for category in spell_categories:
        print(f"\n--- Scraping spells (Category:{category}) ---")
        titles = get_category_members(category)
        print(f"  Found {len(titles)} pages")

        for i, title in enumerate(titles):
            if title in seen:
                continue
            seen.add(title)
            print(f"  [{i+1}/{len(titles)}] {title}", end="")
            time.sleep(DELAY)
            wikitext = get_page_wikitext(title)
            if not wikitext:
                print(" - SKIP")
                continue
            infobox = extract_infobox(wikitext)
            if not infobox:
                print(" - SKIP (no infobox)")
                continue
            infobox_type, raw = infobox
            if "Spell" not in infobox_type:
                print(f" - SKIP (not a spell: {infobox_type})")
                continue
            parsed = parse_infobox(infobox_type, raw)
            parsed["name"] = title
            parsed["wiki_url"] = f"https://masterofmagic.fandom.com/wiki/{title.replace(' ', '_')}"
            all_spells.append(parsed)
            print(f" - OK")

    return all_spells


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Scrape normal units
    normal_units = scrape_category("Normal Units", "Normal Units")
    print(f"\nTotal normal units: {len(normal_units)}")

    # Scrape fantastic creatures
    fantastic = scrape_category("Fantastic Creatures", "Fantastic Creatures")
    print(f"Total fantastic creatures: {len(fantastic)}")

    # Scrape heroes
    heroes = scrape_category("Heroes", "Heroes")
    print(f"Total heroes: {len(heroes)}")

    # Combine all units
    all_units = normal_units + fantastic + heroes
    with open(DATA_DIR / "units.json", "w", encoding="utf-8") as f:
        json.dump(all_units, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_units)} units to data/units.json")

    # Scrape buildings
    buildings = scrape_category("Town Buildings", "Town Buildings")
    with open(DATA_DIR / "buildings.json", "w", encoding="utf-8") as f:
        json.dump(buildings, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(buildings)} buildings to data/buildings.json")

    # Scrape spells
    spells = scrape_spells()
    with open(DATA_DIR / "spells.json", "w", encoding="utf-8") as f:
        json.dump(spells, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(spells)} spells to data/spells.json")

    # Summary
    print("\n=== SCRAPE COMPLETE ===")
    print(f"  Units:     {len(all_units)} ({len(normal_units)} normal, {len(fantastic)} fantastic, {len(heroes)} heroes)")
    print(f"  Buildings: {len(buildings)}")
    print(f"  Spells:    {len(spells)}")


if __name__ == "__main__":
    main()
