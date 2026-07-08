"""Deep debug of Transfermarkt lineup containers."""
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

url = "https://www.transfermarkt.com/brazil_serbia/index/spielbericht/3788846"
r = requests.get(url, headers=headers, timeout=20)

soup = BeautifulSoup(r.text, "html.parser")

# Find ALL divs with aufstellung in class name
for i, box in enumerate(soup.find_all("div", class_=lambda c: c and "aufstellung" in str(c).lower())):
    classes = box.get("class", [])
    print(f"Div {i}: class={' '.join(classes)}")
    text_preview = box.get_text(strip=True)[:100]
    print(f"  Preview: {text_preview}")
    
    # Check for team name heading
    team_header = box.find_previous(["h2", "h3", "strong"])
    if team_header:
        print(f"  Previous header: {team_header.get_text(strip=True)}")
    
    # Count player profile links
    player_links = box.find_all("a", href=lambda h: h and "/profil/spieler/" in (h or ""))
    print(f"  Player links: {len(player_links)}")
    
    # Check for aufstellung-vereinsseite (starting lineup section)
    lineup_section = box.find("div", class_="aufstellung-vereinsseite")
    print(f"  Has lineup section: {lineup_section is not None}")
    
    print()

# Also look at the large-6 columns structure
print("=== Large-6 columns ===")
for i, col in enumerate(soup.find_all("div", class_=lambda c: c and "large-6" in str(c).split())):
    classes = col.get("class", [])
    text_preview = col.get_text(strip=True)[:200]
    print(f"Col {i}: class={' '.join(classes)}")
    print(f"  Preview: {text_preview[:150]}")
    
    # Find team name
    team_name_tag = col.find(["h2", "h3", "strong", "span"])
    if team_name_tag:
        print(f"  Team name tag: {team_name_tag.get_text(strip=True)}")
    
    player_links = col.find_all("a", href=lambda h: h and "/profil/spieler/" in (h or ""))
    print(f"  Players: {len(player_links)}")
    
    # Check for formation subtitle
    formation = col.find("div", class_="formation-subtitle")
    has_starting = "Starting Line-up" in col.get_text() if formation else False
    print(f"  Has formation: {formation is not None}, Starting: {has_starting}")
    print()
