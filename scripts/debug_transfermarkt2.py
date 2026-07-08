"""Debug Transfermarkt with different approaches."""
import requests
from bs4 import BeautifulSoup

urls = [
    ("EN default", "https://www.transfermarkt.com/england/kader/verein/3454", {}),
    ("DE locale", "https://www.transfermarkt.de/england/kader/verein/3454", {"Accept-Language": "de-DE,de;q=0.9"}),
    ("EN fresh", "https://www.transfermarkt.com/england/startseite/verein/3454", {}),
]

base_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

for label, url, extra_headers in urls:
    headers = {**base_headers, **extra_headers}
    r = requests.get(url, headers=headers, timeout=20)
    print(f"\n{'='*60}")
    print(f"  {label} — {url}")
    print(f"  Status: {r.status_code}")
    
    # Check for value-related content
    has_euro = "€" in r.text
    has_market = "Marktwert" in r.text or "market value" in r.text.lower()
    has_table = "items" in r.text
    
    # Check for AJAX/JS loading patterns
    has_xhr = "ajax" in r.text.lower() or "xhr" in r.text.lower()
    has_json_data = "data-value" in r.text or "data-market" in r.text
    
    print(f"  Has €: {has_euro}")
    print(f"  Has market value text: {has_market}")
    print(f"  Has items table: {has_table}")
    print(f"  Has xhr/ajax: {has_xhr}")
    print(f"  Has data-* attributes: {has_json_data}")
    
    # Find any € values in the page
    if has_euro:
        import re
        euro_values = re.findall(r'€[\d.,]+[mMbBkK]?', r.text)
        print(f"  € values found: {euro_values[:10]}")
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Check for specific elements
    rp_tables = soup.find_all("div", class_="responsive-table")
    print(f"  responsive-table divs: {len(rp_tables)}")
    
    items_tables = soup.find_all("table", class_="items")
    print(f"  items tables: {len(items_tables)}")
    
    # Check for inline scripts
    scripts = soup.find_all("script")
    has_value_script = False
    for s in scripts:
        if s.string and ("Marktwert" in s.string or "marketValue" in s.string):
            has_value_script = True
            print(f"  Found value in script: {s.string[:200]}")
            break
    
    if not has_value_script:
        print(f"  No value-related scripts found")
    
    # Print the first 2 tr from the table
    if items_tables:
        first_rows = items_tables[0].find_all("tr")[:5]
        for i, tr in enumerate(first_rows):
            cells = tr.find_all(["td", "th"])
            print(f"  Row {i}: {len(cells)} cells")
            for j, c in enumerate(cells):
                text = c.get_text(strip=True)[:30]
                has_euro_cell = "€" in str(c)
                print(f"    Col {j}: '{text}' {'[€]' if has_euro_cell else ''}")
