import csv
import os
import time

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def accept_cookies(driver):
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        btn.click()
        time.sleep(1)
        return True
    except:
        return False


def extract_matches_from_page(driver):
    rows = driver.find_elements(By.CSS_SELECTOR, ".eventRow")
    matches = []
    current_tournament = None
    current_date = ""

    for row in rows:
        tourn_header = row.find_elements(
            By.CSS_SELECTOR, "[data-testid='header-tournament-item']"
        )
        if tourn_header:
            current_tournament = tourn_header[0].text.strip()

        if not (current_tournament and "superettan" in current_tournament.lower()):
            continue

        date_els = row.find_elements(By.CSS_SELECTOR, "[data-testid='date-header']")
        if date_els:
            current_date = date_els[0].text.strip()

        game_row = row.find_elements(By.CSS_SELECTOR, "[data-testid='game-row']")
        if not game_row:
            continue
        game_row = game_row[0]

        status_els = game_row.find_elements(
            By.CSS_SELECTOR, "[data-testid='time-item'] p"
        )
        status = status_els[0].text.strip() if status_els else ""

        names = game_row.find_elements(By.CSS_SELECTOR, ".participant-name")
        if len(names) < 2:
            continue
        home_team = names[0].text.strip()
        away_team = names[1].text.strip()

        home_score = -1
        away_score = -1
        try:
            separator = game_row.find_element(
                By.XPATH,
                ".//div[contains(@class, 'relative')]"
                "[.//div[contains(@class, 'font-bold')]]",
            )
            parts = separator.text.strip().split("\n")
            if len(parts) >= 3:
                home_score = int(parts[0])
                away_score = int(parts[2])
        except:
            pass

        odds_containers = game_row.find_elements(
            By.XPATH, "./div[contains(@data-testid, 'odd-container')]"
        )
        odds = []
        for oc in odds_containers[:3]:
            try:
                val = oc.text.strip()
                float(val)
                odds.append(val)
            except:
                odds.append("")
        while len(odds) < 3:
            odds.append("")

        matches.append(
            {
                "date": current_date,
                "tournament": current_tournament,
                "status": status,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "odds_1": odds[0],
                "odds_x": odds[1],
                "odds_2": odds[2],
            }
        )

    return matches


def scrape_season(driver, year, max_pages=20):
    url = f"https://www.oddsportal.com/football/sweden/superettan-{year}/results/"
    print(f"  {url}")
    driver.get(url)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".eventRow"))
        )
    except TimeoutException:
        print("    No event rows found")
        return []
    accept_cookies(driver)
    time.sleep(2)

    all_matches = []
    page_num = 1

    while page_num <= max_pages:
        matches = extract_matches_from_page(driver)
        print(f"    Page {page_num}: {len(matches)} matches")
        all_matches.extend(matches)

        try:
            next_link = driver.find_element(By.XPATH, "//a[text()='Next']")
            cls = next_link.get_attribute("class") or ""
            if "disabled" in cls:
                break
            driver.execute_script("arguments[0].click();", next_link)
            time.sleep(3)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".eventRow"))
            )
            page_num += 1
        except NoSuchElementException:
            break

    print(f"    Total: {len(all_matches)} matches")
    return all_matches


def scrape_all(years=None):
    if years is None:
        years = range(2026, 1999, -1)
    driver = create_driver()
    all_matches = []
    try:
        for year in years:
            print(f"\nSeason {year}...")
            all_matches.extend(scrape_season(driver, year))
    finally:
        driver.quit()
    return all_matches


def save_csv(matches, output_path):
    fieldnames = [
        "date",
        "tournament",
        "status",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "odds_1",
        "odds_x",
        "odds_2",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(matches)
    print(f"\nSaved {len(matches)} matches to {output_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", default="oddsportal_se1_historical.csv")
    p.add_argument("--years", type=int, nargs="+")
    p.add_argument("--max-pages", type=int, default=20)
    args = p.parse_args()

    years = args.years or range(2026, 1999, -1)
    matches = scrape_all(years)
    out = os.path.join(os.path.dirname(__file__) or ".", args.output)
    save_csv(matches, out)
