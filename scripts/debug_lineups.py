"""Debug Transfermarkt match report HTML structure for lineups."""
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

url = "https://www.transfermarkt.com/brazil_serbia/index/spielbericht/3788846"
r = requests.get(url, headers=headers, timeout=20)

soup = BeautifulSoup(r.text, "html.parser")

# Find all elements containing "Starting Line-up" or "Line-Ups"
for tag in soup.find_all(["h2", "h3", "strong", "b", "div"]):
    text = tag.get_text(strip=True)
    if "Starting Line-up" in text or "Line-Ups" in text or "Substitutes" in text:
        print(f"Tag: <{tag.name}> class={tag.get('class','')}")
        print(f"  Text: {text[:100]}")
        # Get parent div structure
        parent = tag.parent
        if parent and parent.name == "div":
            print(f"  Parent div class: {parent.get('class', '')}")
            # Get all text from parent
            all_text = parent.get_text("\n")
            lines = [l.strip() for l in all_text.split("\n") if l.strip()]
            print(f"  Lines count: {len(lines)}")
            for line in lines[:15]:
                print(f"  -> {line}")
        print()

# Also look for the lineup table
for table in soup.find_all("table"):
    cls = table.get("class", [])
    if any("lineup" in str(c).lower() for c in cls):
        print(f"Lineup table class: {cls}")
        print(table.get_text()[:500])
