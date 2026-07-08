"""Debug Transfermarkt HTML parsing for problematic teams."""
import requests
from bs4 import BeautifulSoup

base = "https://www.transfermarkt.com"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

teams = {
    "England": "england/kader/verein/3454",
    "France": "frankreich/kader/verein/3455",
    "Germany": "deutschland/kader/verein/3456",
    "Netherlands": "niederlande/kader/verein/3466",
    "Spain": "spanien/kader/verein/3462",
}

for team, path in teams.items():
    url = f"{base}/{path}"
    print(f"\n{'='*60}")
    print(f"  {team} — {url}")
    print(f"{'='*60}")
    
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    
    table = soup.find("table", class_="items")
    if table is None:
        print("  [X] No table with class 'items' found!")
        tables = soup.find_all("table")
        print(f"  Found {len(tables)} tables total")
        for i, t in enumerate(tables):
            cls = t.get("class", [])
            print(f"    Table {i}: class={cls}")
        continue
    
    tbody = table.find("tbody")
    if tbody is None:
        tbody = table
    
    rows = tbody.find_all("tr", recursive=False)
    print(f"  Rows: {len(rows)}")
    
    player_count = 0
    value_zero_count = 0
    
    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue
        
        num_text = cells[0].get_text(strip=True)
        if not num_text.isdigit():
            continue
        
        player_count += 1
        
        name_link = cells[1].find("a")
        player_name = name_link.get_text(strip=True) if name_link else "?"
        
        value_raw = cells[-1].get_text(strip=True) if len(cells) >= 9 else cells[-1].get_text(strip=True)
        value_alt = cells[7].get_text(strip=True) if len(cells) > 7 else ""
        has_euro = "€" in value_raw
        
        was_zero = value_raw in ("-", "—", "", "0") or "€0" in value_raw
        
        if was_zero:
            value_zero_count += 1
        
        if player_count <= 3 or was_zero:
            print(f"  [{player_count:2d}] {player_name:<25} raw='{value_raw[:20]:<20}' "
                  f"alt='{value_alt[:20]:<20}' n_cells={len(cells)} has_eur={has_euro}")
    
    print(f"  Total players: {player_count}, Zero-value: {value_zero_count}")
    
    if rows:
        sample_cells = rows[2].find_all("td")  # third data row
        print(f"  Sample row columns ({len(sample_cells)} cells):")
        for i, c in enumerate(sample_cells):
            text = c.get_text(strip=True)[:50]
            print(f"    Col {i}: '{text}'")
