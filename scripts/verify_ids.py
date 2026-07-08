"""Verify Transfermarkt national team IDs by fetching squad pages."""
import requests
from bs4 import BeautifulSoup

base = "https://www.transfermarkt.com"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

# Teams to check: old ID vs possible correct ID
checks = [
    # (team_name, old_id, new_id)
    ("Brazil", 3439, None),  # old ID seemed correct (€928m)
    ("Argentina", 3437, None),  # old ID seemed correct (€807m)
    ("USA", 3505, None),  # old ID seemed correct (€385m)
    ("Senegal", 3490, None),  # old ID gave 28 players but €0 value
    ("Australia", 3438, None),  # old ID gave 27 players, €209m - seems correct?
    ("Japan", 3460, None),  # old ID gave only 3 players
    ("Belgium", 21, 3383),  # old ID=21 seems wrong (club?)
    ("Croatia", 3448, None),  # old ID gave 33 players, €242m - seems correct
]

for team, old_id, new_id in checks:
    print(f"\n{'='*60}")
    print(f"  {team}")
    print(f"{'='*60}")
    
    ids_to_check = [old_id]
    if new_id:
        ids_to_check.append(new_id)
    
    for tm_id in ids_to_check:
        # Try both English and German slugs
        url = f"{base}/{team.lower()}/kader/verein/{tm_id}"
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else "?"
        
        # Check if it's a national team
        is_national = "National team" in r.text or "Nationalmannschaft" in r.text
        
        # Count players with real values
        table = soup.find("table", class_="items")
        player_count = 0
        value_count = 0
        if table:
            tbody = table.find("tbody") or table
            rows = tbody.find_all("tr", recursive=False)
            for tr in rows:
                cells = tr.find_all("td")
                if len(cells) < 3:
                    continue
                text = tr.get_text()
                if "€" in text:
                    value_count += 1
                player_count += 1
        
        print(f"  ID {tm_id:5d}: status={r.status_code} national={is_national} "
              f"players={player_count} values={value_count} title='{title[:60]}'")
