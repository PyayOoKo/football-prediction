"""Debug Transfermarkt lineup container structure."""
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

url = "https://www.transfermarkt.com/brazil_serbia/index/spielbericht/3788846"
r = requests.get(url, headers=headers, timeout=20)

soup = BeautifulSoup(r.text, "html.parser")

# Find all aufstellung-box containers
for i, box in enumerate(soup.find_all("div", class_="aufstellung-box")):
    print(f"=== Box {i + 1} ===")
    
    # Get the full text
    full_text = box.get_text("\n", strip=True)
    
    # Look for the team name (first line or heading)
    # Find the h2 or strong element with team name
    headings = box.find_all(["h2", "h3", "strong", "a", "span"])
    for h in headings:
        ht = h.get_text(strip=True)
        if ht and 2 < len(ht) < 35 and "Starting" not in ht and "Line-up" not in ht and "Formation" not in ht:
            print(f"  Team: {ht}")
            break
    
    # Find the starting lineup section
    # Look for divs with formation-subtitle
    formations = box.find_all("div", class_="formation-subtitle")
    for f in formations:
        ft = f.get_text(strip=True)
        print(f"  Formation: {ft}")
    
    # Extract all player names - they're in <a> tags or spans with player names
    player_links = box.find_all("a")
    player_names = []
    for a in player_links:
        href = a.get("href", "")
        if "/profil/spieler/" in href:  # Player profile links
            name = a.get_text(strip=True)
            if name and len(name) > 1:
                player_names.append(name)
    
    print(f"  Player links found: {len(player_names)}")
    for pn in player_names:
        print(f"    - {pn}")
    
    print()
