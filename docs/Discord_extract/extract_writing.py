"""
Discord Writing Extractor — v2
Extracts lore, roleplay, and narrative writing from a Discord data package ZIP.
Three channel tiers: LORE (world notes, low threshold), RP (prose, 200+ chars), SESSION (in-play).
Everything else is skipped.
"""

import zipfile
import json
import sys
import os
from datetime import datetime

ZIP_PATH = "package.zip"
OUT_MD = "lore_writing.md"
OUT_JSON = "lore_writing.json"

# ── Channel classification ───────────────────────────────────────────────────

# LORE channels: location notes, world-building, item lists, session recaps.
# Include all messages >= 50 chars.
LORE_CHANNELS = {
    # Campaign 1 — location channels
    "c746582722385805414": "Lore: Aberdeen",
    "c751648260002676767": "Lore: Adelanto",
    "c744033990343786556": "Lore: Elmshire",
    "c749111211374674090": "Lore: Siadiff",
    "c749111544658395207": "Lore: Diondria",
    "c749111725999128656": "Lore: Borilavat",
    "c751647981903806516": "Lore: Finreld",
    "c756287483591000169": "Lore: Handelëtaré",
    "c756714425431621732": "Lore: Zaudfast",
    "c764356191861604385": "Lore: The Elantir Forest",
    "c790292968631828540": "Lore: Felanor",
    "c807480673823883314": "Lore: Morvir",
    "c808475020128813056": "Lore: Zanzarite",
    "c833049403085291580": "Lore: The Slaerian Desert",
    "c850802545881382942": "Lore: Brotor Mountains",
    "c853321362964480010": "Lore: Anearios",
    "c883700703997071370": "Lore: Rellond",
    "c883701377883312170": "Lore: Tianyi Island",
    "c893707447829680138": "Lore: Plane of Air",
    "c908412741105123338": "Lore: The King's Castle",
    "c908415927593353278": "Lore: The Frostlands",
    "c911479203847880735": "Lore: Doragon",
    "c749110949998362684": "Lore: Emyelone",
    "c759251623120994314": "Lore: Nagisa",
    "c779550275743645708": "Lore: Your Ship",
    "c735317621288861789": "Lore: The Village of Sharnwick",
    # Campaign 1 — session recaps, snippets, rules, loot
    "c741776230587695124": "Campaign 1: Session Recaps",
    "c827753724377563227": "Campaign 1: Snippets",
    "c735352064837943426": "Campaign 1: Campaign Guidelines",
    "c735318314309386372": "Campaign 1: Character Sheets",
    "c833039992988631040": "Campaign 1: Loot",
    # Campaign 2
    "c1064388557235556382": "Campaign 2: Session Recaps",
    "c1067556952600760430": "Campaign 2: Snippets",
    "c1064383544119541851": "Campaign 2: Guidelines & Rules",
    "c1064383452545306705": "Campaign 2: Character Sheets",
    "c1064388345570017360": "Campaign 2: Loot",
    # Campaign 3
    "c1360397757541781694": "Campaign 3: Session Recaps",
}

# RP channels: active roleplay prose — include messages >= 200 chars.
RP_CHANNELS = {
    "c804907815951925280": "RP: beardedboggan DM (deleted_user_d12856dcbd03)",
    "c746576429470187632": "RP: Partner 2 DM",
    "c753784860845277264": "RP: Worldbuilding/Lore DM",
    "c852155063790993438": "RP: Campaign Blurbs DM",
    "c1168632061041574089": "RP: Collaborative Group DM",
    "c829074202010320905": "RP: Prose & Poetry",
}

# SESSION channels: in-play notes/actions — include messages >= 80 chars.
SESSION_CHANNELS = {
    "c1049845600155340901": "Campaign 2: Session 1 Party",
    "c1051308915104698408": "Campaign 2: Session 2 Party",
    "c1055635023639887933": "Campaign 2: Session 3 Party",
    "c1067591620050030592": "Campaign 2: Session 4 Party",
    "c1069789708319735808": "Campaign 2: Session 5 Party",
    "c1075620904584888421": "Campaign 2: Session 6 Party",
    "c1077344847532404907": "Campaign 2: Session 7 Party",
    "c1079923692202622986": "Campaign 2: Session 8 Party",
    "c1082813839244021831": "Campaign 2: Session 9 Party",
    "c1083921389460914258": "Campaign 2: Session 10 Party",
    "c1090068039791284286": "Campaign 2: Session 11 Party",
}

THRESHOLDS = {
    "lore": 50,
    "rp": 200,
    "session": 80,
}

# ────────────────────────────────────────────────────────────────────────────


def classify(ch_id):
    if ch_id in LORE_CHANNELS:
        return "lore", LORE_CHANNELS[ch_id]
    if ch_id in RP_CHANNELS:
        return "rp", RP_CHANNELS[ch_id]
    if ch_id in SESSION_CHANNELS:
        return "session", SESSION_CHANNELS[ch_id]
    return None, None


def is_writing(content: str, min_len: int) -> bool:
    c = content.strip()
    if len(c) < min_len:
        return False
    if c.startswith("http") and " " not in c:
        return False
    if c.startswith("https://cdn.discordapp.com") or c.startswith("https://media.discordapp"):
        return False
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    z = zipfile.ZipFile(ZIP_PATH)
    all_names = z.namelist()
    msg_files = sorted([n for n in all_names if n.endswith("messages.json")])

    all_entries = []
    channel_blocks = []
    stats = []

    for mf in msg_files:
        ch_id = mf.split("/")[1]
        tier, label = classify(ch_id)
        if tier is None:
            continue

        messages = json.loads(z.read(mf))
        if not isinstance(messages, list) or not messages:
            continue

        min_len = THRESHOLDS[tier]
        writing_msgs = [m for m in messages if is_writing(m.get("Contents", ""), min_len)]

        if not writing_msgs:
            continue

        writing_msgs.sort(key=lambda m: m.get("Timestamp", ""))
        date_range = f"{writing_msgs[0]['Timestamp'][:10]} → {writing_msgs[-1]['Timestamp'][:10]}"

        stats.append((len(writing_msgs), tier, label))

        for m in writing_msgs:
            all_entries.append({
                "tier": tier,
                "channel_id": ch_id,
                "channel_label": label,
                "timestamp": m.get("Timestamp", ""),
                "message_id": str(m.get("ID", "")),
                "content": m.get("Contents", "").strip(),
            })

        lines = [
            f"## {label}",
            f"*Tier: {tier} · {len(writing_msgs)} messages · {date_range}*",
            "",
        ]
        for m in writing_msgs:
            ts = m.get("Timestamp", "")[:16]
            content = m.get("Contents", "").strip()
            lines.append(f"### {ts}")
            lines.append(content)
            lines.append("")
        channel_blocks.append((tier, label, "\n".join(lines)))

    # Sort blocks: lore first, then rp, then session
    tier_order = {"lore": 0, "rp": 1, "session": 2}
    channel_blocks.sort(key=lambda x: (tier_order[x[0]], x[1]))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = sum(s[0] for s in stats)
    md_header = f"""# Discord Writing Archive
*Extracted {now} · {total} messages across {len(channel_blocks)} channels*

| Tier | Description |
|------|-------------|
| **lore** | World-building notes, location descriptions, NPCs, items — threshold 50 chars |
| **rp** | Active roleplay prose — threshold 200 chars |
| **session** | In-play session notes and actions — threshold 80 chars |

---

"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_header)
        f.write("\n\n---\n\n".join(block for _, _, block in channel_blocks))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"Done.")
    print(f"  Total writing messages : {total:,}")
    print(f"  Channels               : {len(channel_blocks)}")
    print(f"  Output MD              : {OUT_MD}  ({os.path.getsize(OUT_MD):,} bytes)")
    print(f"  Output JSON            : {OUT_JSON}  ({os.path.getsize(OUT_JSON):,} bytes)")
    print()
    print("Per-channel breakdown:")
    stats.sort(key=lambda x: (tier_order[x[1]], -x[0]))
    for count, tier, label in stats:
        print(f"  {count:4d}  [{tier:7}]  {label}")


if __name__ == "__main__":
    main()
