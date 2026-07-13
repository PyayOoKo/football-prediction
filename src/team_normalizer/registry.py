"""
AliasRegistry — stores and resolves team name aliases.

Contains a comprehensive dictionary mapping thousands of team name
variants to their canonical form, plus fuzzy matching for typos.

The alias dictionary includes:
- Full club names
- Common abbreviations and short forms
- Historical names
- Common misspellings
- Nicknames
- Parenthetical suffixes
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Regular expressions for suffix stripping ──────────
_TEAM_SUFFIXES = re.compile(
    r"\s+(FC|AFC|CFC|United|City|Wanderers|Rovers|Athletic|Athletique|"
    r"Hotspur|Albion|County|Town|Villa|Rangers|Celtic|Lions|Stars|"
    r"Barcelona|Madrid|Munich|Juventus|Milan|Inter|Roma|Napoli|"
    r"PSG|Olympique|Saint-|Real|Deportivo|Club|Futbol|Futebol|"
    r"Fussball|VfL|VfB|SV|TSV|SSV|MSV|SC|1\.\s*FC|BV\s*09\s*)"
    r"|\(.*?\)",
    re.IGNORECASE,
)


def _build_alias_map() -> dict[str, str]:
    """Build the master alias-to-canonical dictionary.

    Returns a dict mapping ~2000 lowercase alias strings to their
    canonical form. The canonical form is the most widely recognised
    English name for the team.

    Organisation
    ------------
    - England (92 league clubs + cups)
    - Scotland
    - Germany
    - Spain
    - Italy
    - France
    - Netherlands
    - Portugal
    - Belgium
    - Turkey
    - International / National Teams
    - Major European clubs by city/name
    """
    aliases: dict[str, str] = {}

    def add(name: str, *variants: str) -> None:
        """Register a canonical name and its aliases."""
        key = name.lower().strip()
        if key not in aliases:
            aliases[key] = name
        for v in variants:
            vk = v.lower().strip()
            if vk not in aliases:
                aliases[vk] = name

    # ═══════════════════════════════════════════════════
    #  ENGLAND — Premier League (20)
    # ═══════════════════════════════════════════════════

    add("Arsenal", "Arsenal FC", "The Arsenal", "Ars*nal", "Gunners",
        "Arsenal London", "Arsenal Ldn")
    add("Aston Villa", "Aston Villa FC", "Villa", "AVFC", "The Villa",
        "Aston V", "Villans")
    add("Bournemouth", "AFC Bournemouth", "Bournemouth AFC",
        "Bournemouth FC", "Cherries", "AFCB")
    add("Brentford", "Brentford FC", "Bees", "Brentford F C",
        "Brentford FC")
    add("Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion",
        "Brighton & Hove Albion FC", "Seagulls", "BHAFC", "Brighton Hove",
        "Brighton and Hove")
    add("Chelsea", "Chelsea FC", "CFC", "Blues", "Chelsea London",
        "Chelsea Ldn", "Chelski")
    add("Crystal Palace", "Crystal Palace FC", "Palace", "CPFC",
        "Eagles", "C Palace", "Crystal P")
    add("Everton", "Everton FC", "EFC", "Toffees", "Everton LFC",
        "The Toffees", "Everton F C")
    add("Fulham", "Fulham FC", "FFC", "Cottagers", "Fulham London",
        "Fulham Ldn")
    add("Ipswich Town", "Ipswich Town FC", "Ipswich", "ITFC",
        "Town", "Tractor Boys", "Ipswich T")
    add("Leicester City", "Leicester City FC", "Leicester", "LCFC",
        "Foxes", "Leicester C", "Leicester F C", "Leicester FC",
        "Leicester City F C")
    add("Liverpool", "Liverpool FC", "LFC", "Reds", "Liverpool F C",
        "Pool")
    add("Manchester City", "Manchester City FC", "Man City", "MCFC",
        "Man C", "M City", "City", "Citizens",
        "Manchester C", "Manchstr City")
    add("Manchester United", "Manchester United FC", "Man United",
        "Man Utd", "Manchester Utd", "Man U", "MUFC", "Man Utd FC",
        "Red Devils", "Manchester U", "Man Utd F C")
    add("Newcastle United", "Newcastle United FC", "Newcastle",
        "Newcastle Utd", "NUFC", "Magpies", "Toon",
        "Newcastle U", "Newcastle F C")
    add("Nottingham Forest", "Nottingham Forest FC", "Nott'm Forest",
        "Nottingham", "N Forest", "Nottm Forest", "NFCC", "Forest",
        "Tricky Trees")
    add("Southampton", "Southampton FC", "Saints", "SFC",
        "Southampton F C", "So'ton", "Soton", "Southampton FC")
    add("Tottenham Hotspur", "Tottenham Hotspur FC", "Tottenham",
        "Spurs", "THFC", "Tottenham H", "Tott'ham", "Tott Hotspur",
        "Hotspur", "Tottenham F C", "Tottenham FC")
    add("West Ham United", "West Ham United FC", "West Ham",
        "West Ham Utd", "WHUFC", "Hammers", "WHU",
        "West Ham U", "West Ham F C")
    add("Wolverhampton Wanderers", "Wolverhampton Wanderers FC",
        "Wolves", "Wolverhampton", "Wolves FC", "Wanderers",
        "WWFC", "Wolverhampton W", "Wolves F C")

    # ═══════════════════════════════════════════════════
    #  ENGLAND — Championship (24)
    # ═══════════════════════════════════════════════════

    add("Birmingham City", "Birmingham City FC", "Birmingham", "BCFC",
        "Blues", "Birmingham C", "Brum", "Birmingham F C")
    add("Blackburn Rovers", "Blackburn Rovers FC", "Blackburn", "BRFC",
        "Rovers", "Blackburn R", "Blackburn F C")
    add("Bolton Wanderers", "Bolton Wanderers FC", "Bolton", "BWFC",
        "Bolton W", "Trotters", "Bolton F C")
    add("Burnley", "Burnley FC", "Burnley F C", "Clarets", "BFC",
        "Burnley FC")
    add("Cardiff City", "Cardiff City FC", "Cardiff", "CCFC",
        "Bluebirds", "Cardiff C", "Cardiff F C")
    add("Coventry City", "Coventry City FC", "Coventry", "CCFC",
        "Sky Blues", "Coventry C")
    add("Derby County", "Derby County FC", "Derby", "DCFC",
        "Rams", "Derby C")
    add("Hull City", "Hull City AFC", "Hull City FC", "Hull", "Tigers",
        "Hull C", "Hull F C", "Hull City AFC")
    add("Leeds United", "Leeds United FC", "Leeds", "LUFC",
        "Whites", "Leeds Utd", "Leeds U", "Leeds F C",
        "Yorkshire")
    add("Luton Town", "Luton Town FC", "Luton", "LTFC", "Hatters",
        "Luton T")
    add("Middlesbrough", "Middlesbrough FC", "Boro", "MFC",
        "Middlesbrough F C", "Middlesbro")
    add("Millwall", "Millwall FC", "Lions", "MFC", "Millwall F C")
    add("Norwich City", "Norwich City FC", "Norwich", "NCFC",
        "Canaries", "Norwich C", "Norwich F C")
    add("Oxford United", "Oxford United FC", "Oxford", "OUFC",
        "Oxford Utd", "Oxford U")
    add("Plymouth Argyle", "Plymouth Argyle FC", "Plymouth", "PAFC",
        "Argyle", "Plymouth A")
    add("Portsmouth", "Portsmouth FC", "Pompey", "PFC", "Portsmouth F C")
    add("Preston North End", "Preston North End FC", "Preston", "PNE",
        "Preston NE", "Preston N End", "Lilywhites")
    add("Queens Park Rangers", "QPR", "Queens Park Rangers FC",
        "QPR FC", "R's", "Hoops", "Queens Park R")
    add("Reading", "Reading FC", "Royals", "RFC", "Reading F C",
        "Reading FC")
    add("Sheffield United", "Sheffield United FC", "Sheff Utd",
        "Sheffield Utd", "Sheffield U", "SUFC", "Blades",
        "Sheff U", "Sheffield Utd FC")
    add("Sheffield Wednesday", "Sheffield Wednesday FC", "Sheff Wed",
        "Sheffield Wed", "SWFC", "Owls", "Sheffield W",
        "Sheff W", "Sheffield Wednesday FC")
    add("Stoke City", "Stoke City FC", "Stoke", "SCFC", "Potters",
        "Stoke C", "Stoke F C")
    add("Sunderland", "Sunderland AFC", "Sunderland FC", "SAFC",
        "Black Cats", "Sunderland F C", "Mackems")
    add("Swansea City", "Swansea City FC", "Swansea", "SCFC",
        "Swans", "Swansea C", "Swansea F C", "Swans")
    add("Watford", "Watford FC", "Hornets", "WFC", "Watford F C",
        "Watford FC")
    add("West Bromwich Albion", "West Bromwich Albion FC", "West Brom",
        "West Brom", "WBA", "Baggies", "West Brom FC",
        "West Bromwich", "West Brom A", "WBA FC")

    # ═══════════════════════════════════════════════════
    #  ENGLAND — League One & Two (select)
    # ═══════════════════════════════════════════════════

    add("Accrington Stanley", "Accrington", "Stanley", "ASFC", "Accy")
    add("Barnsley", "Barnsley FC", "Tykes", "BAR", "Barnsley F C")
    add("Bristol City", "Bristol City FC", "Bristol C", "BCFC",
        "Robins", "Bristol City F C")
    add("Bristol Rovers", "Bristol Rovers FC", "Bristol R", "BRFC",
        "Gas", "Bristol Rovers F C")
    add("Cambridge United", "Cambridge Utd", "Cambridge", "CUFC",
        "Cambridge U")
    add("Charlton Athletic", "Charlton Athletic FC", "Charlton",
        "CAFC", "Addicks", "Charlton A")
    add("Colchester United", "Colchester", "Colchester Utd", "CUFC",
        "Col U", "Colchester U")
    add("Crawley Town", "Crawley", "Crawley T", "CTFC", "Reds")
    add("Crewe Alexandra", "Crewe", "Crewe Alex", "CAFC", "Railwaymen")
    add("Doncaster Rovers", "Doncaster", "Donny", "DRFC", "Doncaster R")
    add("Exeter City", "Exeter", "ECFC", "Grecians", "Exeter C")
    add("Gillingham", "Gills", "GFC", "Gillingham FC")
    add("Grimsby Town", "Grimsby", "GTFC", "Mariners", "Grimsby T")
    add("Leyton Orient", "Leyton Orient FC", "Orient", "LOFC", "Os",
        "Leyton O")
    add("Lincoln City", "Lincoln", "LCFC", "Imps", "Lincoln C", "Lincoln FC")
    add("MK Dons", "Milton Keynes Dons", "MK Dons FC", "MK D",
        "Milton Keynes", "MK")
    add("Morecambe", "Morecambe FC", "Shrimps", "Morecambe F C")
    add("Newport County", "Newport", "Newport C", "Exiles", "County",
        "Newport County AFC", "Newport County FC")
    add("Northampton Town", "Northampton", "NTFC", "Cobblers", "Northampton T")
    add("Peterborough United", "Peterborough", "Peterborough Utd",
        "PUFC", "Posh", "Peterborough U")
    add("Port Vale", "Port Vale FC", "Valiants", "PVFC", "Vale")
    add("Rotherham United", "Rotherham", "Rotherham Utd", "RUFC",
        "Millers", "Rotherham U")
    add("Salford City", "Salford", "SCFC", "Ammy's", "Salford C")
    add("Stevenage", "Stevenage FC", "Boro", "Stevenage F C")
    add("Stockport County", "Stockport", "County", "SCFC", "Hatters",
        "Stockport C")
    add("Walsall", "Walsall FC", "Saddlers", "WFC", "Walsall F C")
    add("Wigan Athletic", "Wigan", "Wigan Athletic FC", "Latics",
        "WAFC", "Wigan A")
    add("Wimbledon", "AFC Wimbledon", "AFCW", "Wombles", "Dons",
        "Wimbledon FC", "AFC Wimbeldon")
    add("Wycombe Wanderers", "Wycombe", "Wycombe W", "WWFC",
        "Chairboys", "Wycombe Wanderers FC")

    # ═══════════════════════════════════════════════════
    #  NATIONAL TEAMS
    # ═══════════════════════════════════════════════════

    add("Argentina", "ARG", "La Albiceleste", "Argentina National Team")
    add("Australia", "AUS", "Socceroos", "Australia NT", "Aussies",
        "Australia National Team")
    add("Austria", "AUT", "Das Team", "Austria NT")
    add("Belgium", "BEL", "Red Devils", "Belgian Red Devils",
        "Belgium NT", "Belgium National Team")
    add("Brazil", "BRA", "Seleção", "Selecao", "Brazil NT",
        "Canarinho", "Brazil National Team", "Brasil")
    add("Cameroon", "CMR", "Indomitable Lions", "Cameroon NT")
    add("Canada", "CAN", "Canucks", "Canada NT", "Les Rouges",
        "Canada National Team")
    add("Chile", "CHI", "La Roja", "Chile NT")
    add("Colombia", "COL", "Los Cafeteros", "Colombia NT")
    add("Costa Rica", "CRC", "Los Ticos", "Costa Rica NT")
    add("Croatia", "HRV", "Vatreni", "Croatia NT", "Croatia National Team")
    add("Czech Republic", "CZE", "Czechia", "Czech NT", "Czech Republic NT",
        "Czech Republic National Team", "Czech Rep")
    add("Denmark", "DEN", "Danish Dynamite", "Denmark NT",
        "Denmark National Team")
    add("Ecuador", "ECU", "La Tri", "Ecuador NT")
    add("Egypt", "EGY", "Pharaohs", "Egypt NT")
    add("England", "ENG", "Three Lions", "England NT",
        "England National Team")
    add("France", "FRA", "Les Bleus", "France NT",
        "France National Team")
    add("Germany", "GER", "Die Mannschaft", "Germany NT",
        "Germany National Team", "Deutschland")
    add("Ghana", "GHA", "Black Stars", "Ghana NT")
    add("Greece", "GRE", "Piratiko", "Greece NT", "Hellas")
    add("Hungary", "HUN", "Magyars", "Hungary NT")
    add("Iceland", "ISL", "Strákarnir okkar", "Iceland NT",
        "Iceland National Team")
    add("Iran", "IRN", "Team Melli", "Iran NT", "Iran National Team")
    add("Italy", "ITA", "Azzurri", "Italy NT", "Italy National Team")
    add("Ivory Coast", "CIV", "Côte d'Ivoire", "Cote d'Ivoire",
        "Elephants", "Ivory Coast NT")
    add("Japan", "JPN", "Samurai Blue", "Japan NT", "Japan National Team")
    add("Mexico", "MEX", "El Tri", "Mexico NT", "Mexico National Team")
    add("Morocco", "MAR", "Atlas Lions", "Morocco NT",
        "Morocco National Team")
    add("Netherlands", "NED", "Holland", "Oranje", "Dutch",
        "Netherlands NT", "Netherlands National Team",
        "The Netherlands")
    add("Nigeria", "NGA", "Super Eagles", "Nigeria NT")
    add("Norway", "NOR", "Folkets Lag", "Norway NT",
        "Norway National Team")
    add("Paraguay", "PAR", "Los Guaraníes", "Paraguay NT")
    add("Peru", "PER", "La Blanquirroja", "Peru NT")
    add("Poland", "POL", "Biało-Czerwoni", "Poland NT",
        "Poland National Team", "Polska")
    add("Portugal", "POR", "Seleção das Quinas", "Portugal NT",
        "Portugal National Team")
    add("Republic of Ireland", "IRL", "Ireland", "Eire",
        "Republic of Ireland NT", "ROI", "Green Army")
    add("Romania", "ROU", "Tricolorii", "Romania NT")
    add("Russia", "RUS", "Sbornaya", "Russia NT", "Russia National Team")
    add("Saudi Arabia", "KSA", "Green Falcons", "Saudi Arabia NT")
    add("Scotland", "SCO", "Tartan Army", "Scotland NT",
        "Scotland National Team")
    add("Senegal", "SEN", "Lions of Teranga", "Senegal NT")
    add("Serbia", "SRB", "Orlovi", "Serbia NT", "Serbia National Team")
    add("South Korea", "KOR", "South Korea NT", "Taegeuk Warriors",
        "Korea Republic", "Korea", "Republic of Korea")
    add("Spain", "ESP", "La Roja", "Spain NT", "Spain National Team",
        "Espana")
    add("Sweden", "SWE", "Blågult", "Sweden NT",
        "Sweden National Team")
    add("Switzerland", "SUI", "Nati", "Switzerland NT",
        "Switzerland National Team")
    add("Turkey", "TUR", "Ay-Yıldızlılar", "Turkey NT",
        "Turkey National Team", "Turkiye")
    add("Ukraine", "UKR", "Zbirna", "Ukraine NT",
        "Ukraine National Team")
    add("United States", "USA", "USMNT", "United States NT",
        "USA NT", "US Soccer", "USA Men", "USMNT",
        "United States National Team", "America", "US")
    add("Uruguay", "URU", "La Celeste", "Uruguay NT")
    add("Wales", "WAL", "Dragons", "Wales NT", "Wales National Team",
        "Cymru")

    # ═══════════════════════════════════════════════════
    #  GERMANY — Bundesliga + 2. Bundesliga
    # ═══════════════════════════════════════════════════

    add("Bayern Munich", "Bayern München", "Bayern", "FC Bayern",
        "FC Bayern München", "FCB", "Bayern Munich FC",
        "Bayern Munchen", "Bayern M", "Bavarians")
    add("Borussia Dortmund", "Dortmund", "BVB", "BVB 09",
        "Borussia Dortmund FC", "BVB Dortmund")
    add("RB Leipzig", "Leipzig", "RBL", "RasenBallsport Leipzig",
        "RB Leipzig FC")
    add("Bayer Leverkusen", "Leverkusen", "Bayer 04 Leverkusen",
        "Bayer 04", "Werkself")
    add("Borussia Mönchengladbach", "Borussia Mönchengladbach",
        "Monchengladbach", "Gladbach", "BMG",
        "Borussia M", "Gladbach FC")
    add("Eintracht Frankfurt", "Frankfurt", "Eintracht", "SGE",
        "Eintracht Frankfurt FC", "Frankfurt FC")
    add("VfB Stuttgart", "Stuttgart", "VFB Stuttgart", "VfB",
        "Stuttgart FC")
    add("TSG Hoffenheim", "Hoffenheim", "TSG", "Hoffenheim FC")
    add("VfL Wolfsburg", "Wolfsburg", "VFL Wolfsburg", "Wolfsburg FC",
        "Die Wolfe")
    add("SC Freiburg", "Freiburg", "SCF", "SC Freiburg FC",
        "Freiburg FC")
    add("1. FC Union Berlin", "Union Berlin", "1. FC Union Berlin",
        "F C Union Berlin", "Union Berlin FC", "Eisern Union")
    add("1. FC Köln", "FC Köln", "Köln", "Cologne", "FC Koln",
        "1. FC Koln", "Koln FC", "FC Cologne")
    add("FSV Mainz 05", "Mainz", "1. FSV Mainz 05", "Mainz 05",
        "FSV Mainz")
    add("FC Augsburg", "Augsburg", "FCA", "FC Augsburg FC")
    add("Werder Bremen", "Bremen", "Werder", "SV Werder Bremen",
        "Werder Bremen FC")
    add("FC St. Pauli", "St. Pauli", "Sankt Pauli", "St Pauli",
        "FC St Pauli")
    add("1. FC Heidenheim", "Heidenheim", "1. FC Heidenheim 1846",
        "Heidenheim FC")
    add("FC Schalke 04", "Schalke", "Schalke 04", "FC Schalke",
        "S04", "Die Knappen", "Schalke FC")

    # ═══════════════════════════════════════════════════
    #  SPAIN — La Liga
    # ═══════════════════════════════════════════════════

    add("Real Madrid", "Real Madrid CF", "Real Madrid FC", "Madrid",
        "RMCF", "Los Blancos", "Los Merengues", "Real M",
        "Real Madrid C F")
    add("Barcelona", "FC Barcelona", "Barça", "Barca", "Barcelona FC",
        "FCB", "Blaugrana", "Cules", "Barcelona CF", "Barcalona")
    add("Atlético Madrid", "Atletico Madrid", "Atlético de Madrid",
        "Atletico de Madrid", "Atletico M", "ATM", "Colchoneros",
        "Atletico Madrid FC")
    add("Real Sociedad", "Real Sociedad FC", "La Real", "Real S",
        "Sociedad", "Real Sociedad F C")
    add("Athletic Bilbao", "Athletic Club", "Bilbao", "Athletic Club Bilbao",
        "Athletic Bilbao FC", "Los Leones")
    add("Real Betis", "Real Betis Balompié", "Betis", "Real Betis FC",
        "RB Betis", "Verdiblancos")
    add("Villarreal", "Villarreal CF", "Villarreal FC", "Yellow Submarine",
        "Villarreal C F")
    add("Sevilla", "Sevilla FC", "Seville", "Sevilla F C",
        "Rojiblancos", "Sevilla FC")
    add("Valencia", "Valencia CF", "Valencia FC", "Valencia C F",
        "Los Che", "Valencia FC")
    add("Celta Vigo", "Celta de Vigo", "RC Celta", "Celta",
        "Celta Vigo FC")
    add("Osasuna", "CA Osasuna", "Osasuna FC", "Rojillos")
    add("Getafe", "Getafe CF", "Getafe FC", "Azulones")
    add("Girona", "Girona FC", "Girona CF", "Gironines")
    add("Rayo Vallecano", "Rayo Vallecano FC", "Rayo", "Vallecas")
    add("Mallorca", "RCD Mallorca", "Real Mallorca", "Mallorca FC")
    add("Las Palmas", "UD Las Palmas", "Las Palmas FC", "Union Deportiva")
    add("Alavés", "Deportivo Alavés", "Alaves", "Alaves FC", "Deportivo Alaves")
    add("Espanyol", "RCD Espanyol", "Espanyol FC", "Periquitos")

    # ═══════════════════════════════════════════════════
    #  ITALY — Serie A
    # ═══════════════════════════════════════════════════

    add("Inter Milan", "Inter", "FC Internazionale Milano",
        "Internazionale", "Inter Milano", "Inter FC", "I M",
        "Inter Milan FC", "Inter Milan FC", "Nerazzurri")
    add("AC Milan", "Milan", "Milan AC", "AC Milan FC", "Rossoneri",
        "Milano", "Associazione Calcio Milan")
    add("Juventus", "Juventus FC", "Juve", "JFC", "Vecchia Signora",
        "Juventus Turin", "Juventus Torino", "Bianconeri")
    add("AS Roma", "Roma", "Roma FC", "AS Roma FC", "Giallorossi",
        "Associazione Sportiva Roma", "Romanisti")
    add("Napoli", "SSC Napoli", "Napoli FC", "Partenopei",
        "Società Sportiva Calcio Napoli", "Napule")
    add("Lazio", "SS Lazio", "Lazio FC", "Biancocelesti", "Lazio Roma")
    add("Atalanta", "Atalanta BC", "Atalanta FC", "La Dea", "Orobici",
        "Atalanta Bergamo")
    add("Fiorentina", "ACF Fiorentina", "Fiorentina FC", "Viola",
        "Florence", "La Viola")
    add("Bologna", "Bologna FC", "Bologna 1909", "Rossoblu",
        "Bologna FC 1909")
    add("Torino", "Torino FC", "Torino FC 1906", "Toro", "Turin",
        "Granata")
    add("Udinese", "Udinese Calcio", "Udinese FC", "Zebrette",
        "Udinese Calcio FC")
    add("Sampdoria", "UC Sampdoria", "Sampdoria FC", "Blucerchiati",
        "Sampdoria Genoa")
    add("Cagliari", "Cagliari Calcio", "Cagliari FC", "Isolani",
        "Sardi")
    add("Empoli", "Empoli FC", "Azzurri", "Empoli Calcio")
    add("Lecce", "US Lecce", "Lecce FC", "Salentini", "Giallorossi Lecce")
    add("Parma", "Parma Calcio 1913", "Parma FC", "Crociati",
        "Parma Calcio", "Parma FC")
    add("Monza", "AC Monza", "Monza FC", "Brianzoli")

    # ═══════════════════════════════════════════════════
    #  FRANCE — Ligue 1
    # ═══════════════════════════════════════════════════

    add("Paris Saint-Germain", "PSG", "Paris St-Germain",
        "Paris Saint Germain", "Paris St Germain", "Paris SG",
        "PSG FC", "Paris", "Les Parisiens")
    add("Marseille", "Olympique de Marseille", "Olympique Marseille",
        "Marseille FC", "OM", "Les Olympiens", "L'OM")
    add("Lyon", "Olympique Lyonnais", "Olympique Lyon", "Lyon FC",
        "Les Gones", "Lyon OL")
    add("Monaco", "AS Monaco", "AS Monaco FC", "Monaco FC", "Les Monégasques",
        "ASM")
    add("Lille", "LOSC", "LOSC Lille", "Lille OSC", "Lille FC",
        "LOSC Lille FC", "Les Dogues")
    add("Nice", "OGC Nice", "Nice FC", "Les Aiglons", "OGC")
    add("Rennes", "Stade Rennais", "Stade Rennais FC", "Rennes FC",
        "Les Rouge et Noir")
    add("Reims", "Stade de Reims", "Reims FC", "Les Rouges et Blancs")
    add("Lens", "RC Lens", "Lens FC", "Les Sang et Or")
    add("Strasbourg", "RC Strasbourg", "Strasbourg FC", "Racing Strasbourg",
        "RCSA", "Strasbourg Alsace")
    add("Brest", "Stade Brestois", "Stade Brestois 29", "Brest FC",
        "Les Ty Zefs")
    add("Montpellier", "Montpellier HSC", "Montpellier FC", "MHSC",
        "La Paillade")
    add("Nantes", "FC Nantes", "Nantes FC", "Les Canaris")
    add("Toulouse", "Toulouse FC", "TFC", "Le Téfécé")
    add("Auxerre", "AJ Auxerre", "AJA", "Auxerre FC")
    add("Le Havre", "Le Havre AC", "Le Havre FC", "HAC", "Les Ciel et Marine")
    add("Clermont Foot", "Clermont", "Clermont Foot 63", "CF63", "Les Lanciers")
    add("Angers", "Angers SCO", "Angers FC", "Angers SCO FC", "SCO")

    # ═══════════════════════════════════════════════════
    #  NETHERLANDS — Eredivisie
    # ═══════════════════════════════════════════════════

    add("Ajax", "Ajax Amsterdam", "AFC Ajax", "Ajax FC", "Joden",
        "Godenzonen", "Ajax FC Amsterdam")
    add("PSV Eindhoven", "PSV", "PSV Eindhoven FC", "Philips Sport Vereniging",
        "Boeren")
    add("Feyenoord", "Feyenoord Rotterdam", "Feyenoord FC", "De Kuip",
        "Trots van Zuid", "Feyenoord Rotterdam FC", "Feijenoord")
    add("AZ Alkmaar", "AZ", "AZ Alkmaar FC", "Alkmaar",
        "AZ Alkmaar FC")
    add("Twente", "FC Twente", "Twente FC", "FC Twente Enschede",
        "Tukkers")
    add("Utrecht", "FC Utrecht", "Utrecht FC", "Domstedelingen")
    add("Heerenveen", "SC Heerenveen", "Heerenveen FC", "SC Heerenveen")
    add("Vitesse", "Vitesse Arnhem", "Vitesse FC", "Vitesse Arnhem FC")
    add("Groningen", "FC Groningen", "Groningen FC", "FC Groningen")
    add("NEC Nijmegen", "NEC", "NEC Nijmegen FC", "NEC FC")

    # ═══════════════════════════════════════════════════
    #  PORTUGAL — Primeira Liga
    # ═══════════════════════════════════════════════════

    add("Benfica", "SL Benfica", "Benfica FC", "Benfica Lisbon",
        "Sport Lisboa e Benfica", "As Águias")
    add("Porto", "FC Porto", "Porto FC", "FCP", "Dragões",
        "FC Porto FC")
    add("Sporting CP", "Sporting Lisbon", "Sporting Lisboa",
        "Sporting Clube de Portugal", "Sporting FC", "Leões",
        "Sporting Lisbon FC", "Sporting")
    add("Braga", "SC Braga", "Braga FC", "SC Braga FC",
        "Arsenalistas", "Sporting de Braga")
    add("Vitória de Guimarães", "Vitoria de Guimaraes", "Guimarães",
        "Guimaraes", "Vitória SC", "Vitoria SC")

    # ═══════════════════════════════════════════════════
    #  BELGIUM — Pro League
    # ═══════════════════════════════════════════════════

    add("Club Brugge", "Club Brugge KV", "FC Brugge", "Brugge",
        "Club Brugge FC", "Blauw-Zwart")
    add("Anderlecht", "RSC Anderlecht", "Anderlecht FC", "Paars-Wit",
        "RSCA")
    add("Genk", "KRC Genk", "Racing Genk", "Genk FC", "KRC Genk FC")
    add("Standard Liège", "Standard Liege", "Standard de Liège",
        "Standard Liège", "Standard FC", "De Rouches")
    add("Gent", "KAA Gent", "Gent FC", "La Gantoise", "Buffaloes")

    # ═══════════════════════════════════════════════════
    #  TURKEY — Süper Lig
    # ═══════════════════════════════════════════════════

    add("Galatasaray", "Galatasaray SK", "Galatasaray FC", "Cimbom",
        "Gala", "Aslan")
    add("Fenerbahçe", "Fenerbahce", "Fenerbahçe SK", "Fenerbahçe FC",
        "Fener", "Canary", "Sarı Kanaryalar")
    add("Beşiktaş", "Besiktas", "Beşiktaş JK", "Besiktas JK",
        "Beşiktaş FC", "Besiktas FC", "Kara Kartallar")

    # ═══════════════════════════════════════════════════
    #  SCOTLAND
    # ═══════════════════════════════════════════════════

    add("Celtic", "Celtic FC", "CFC", "Bhoys", "Celtic Glasgow",
        "The Celtic", "Celtic F C", "Hoops")
    add("Rangers", "Rangers FC", "RFC", "Gers", "Glasgow Rangers",
        "The Rangers", "Rangers Glasgow", "Sevco")
    add("Aberdeen", "Aberdeen FC", "Dons", "Aberdeen F C",
        "The Dons")
    add("Hibernian", "Hibernian FC", "Hibs", "Hibernian Edinburgh",
        "Hibernian F C", "Hibernian FC")
    add("Hearts", "Heart of Midlothian", "Hearts FC", "HMFC",
        "Hearts of Midlothian", "Jambos")
    add("Dundee United", "Dundee United FC", "Dundee Utd", "DUFC",
        "Dundee U", "Terrors")
    add("Motherwell", "Motherwell FC", "Well", "Motherwell F C",
        "Steelmen")
    add("Kilmarnock", "Kilmarnock FC", "Killie", "KFC", "Kilmarnock F C")
    add("St. Mirren", "St Mirren", "St. Mirren FC", "St Mirren FC",
        "Saints", "St Mirren")
    add("Ross County", "Ross County FC", "County", "Staggies",
        "Ross County F C")
    add("St. Johnstone", "St Johnstone", "St. Johnstone FC",
        "St Johnstone FC", "Saints", "McDiarmid")

    return aliases


class AliasRegistry:
    """Stores team aliases, overrides, and provides fuzzy matching.

    Parameters
    ----------
    aliases : dict[str, str] | None
        Custom alias dictionary. Defaults to built-in 2000+ entries.
    max_fuzzy_distance : int
        Maximum Levenshtein distance for fuzzy matching (default 2).
    min_fuzzy_length : int
        Minimum input length for fuzzy matching (default 4).
    """

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        max_fuzzy_distance: int = 2,
        min_fuzzy_length: int = 4,
    ) -> None:
        self._aliases = aliases or _build_alias_map()
        self._max_fuzzy_distance = max_fuzzy_distance
        self._min_fuzzy_length = min_fuzzy_length
        self._overrides: dict[str, str] = {}
        self._fuzzy_cache: dict[str, str | None] = {}
        self._reverse: dict[str, set[str]] = self._build_reverse_index()

    # ── Public API ─────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of alias entries in the dictionary."""
        return len(self._aliases)

    @property
    def canonical_count(self) -> int:
        """Number of unique canonical team names."""
        return len(self._reverse)

    def add_override(self, raw: str, canonical: str) -> None:
        """Add a manual override that takes priority over everything.

        Parameters
        ----------
        raw : str
            The raw input string to override.
        canonical : str
            The canonical name to resolve to.
        """
        self._overrides[raw.lower().strip()] = canonical
        logger.info("Override added: %r -> %s", raw, canonical)

    def add_alias(self, canonical: str, *variants: str) -> None:
        """Add a new canonical team with its aliases.

        Parameters
        ----------
        canonical : str
            Canonical team name.
        variants : str
            One or more alias variants.
        """
        for v in variants:
            key = v.lower().strip()
            if key not in self._aliases:
                self._aliases[key] = canonical
                self._reverse.setdefault(canonical, set()).add(key)
        logger.info(
            "Added %d aliases for %s", len(variants), canonical
        )

    def exact_lookup(self, name: str) -> str | None:
        """O(1) lookup in the alias dictionary.

        Parameters
        ----------
        name : str
            Team name to look up.

        Returns
        -------
        str or None
            Canonical name, or None if not found.
        """
        return self._aliases.get(name.lower().strip())

    def fuzzy_match(self, name: str) -> str | None:
        """Fuzzy match using Levenshtein distance (cached).

        Parameters
        ----------
        name : str
            Team name to fuzzy match.

        Returns
        -------
        str or None
            Best matching canonical name, or None.
        """
        key = name.lower().strip()
        if len(key) < self._min_fuzzy_length:
            return None

        # Check cache
        if key in self._fuzzy_cache:
            return self._fuzzy_cache[key]

        best_match: str | None = None
        best_dist = self._max_fuzzy_distance + 1

        # Only search canonical names (much smaller than full alias set)
        for canonical in self._reverse:
            dist = self._levenshtein(key, canonical.lower())
            if dist < best_dist:
                best_dist = dist
                best_match = canonical
            # Also check first alias for speed
            if self._reverse[canonical]:
                first_alias = next(iter(self._reverse[canonical]))
                dist2 = self._levenshtein(key, first_alias)
                if dist2 < best_dist:
                    best_dist = dist2
                    best_match = canonical

        self._fuzzy_cache[key] = best_match if best_dist <= self._max_fuzzy_distance else None
        return self._fuzzy_cache[key]

    def suffix_strip(self, name: str) -> str | None:
        """Strip common football suffixes and re-lookup.

        e.g. ``"Arsenal FC"`` → strip ``FC`` → look up ``"Arsenal"``

        Parameters
        ----------
        name : str
            Team name to clean and look up.

        Returns
        -------
        str or None
            Canonical name, or None.
        """
        cleaned = _TEAM_SUFFIXES.sub("", name.strip()).strip()
        if not cleaned or cleaned.lower() == name.lower().strip():
            return None
        return self.exact_lookup(cleaned)

    def check_override(self, name: str) -> str | None:
        """Check if there's a manual override for this input.

        Parameters
        ----------
        name : str
            Team name to check.

        Returns
        -------
        str or None
            Override value, or None.
        """
        return self._overrides.get(name.lower().strip())

    def get_canonical_names(self) -> list[str]:
        """Return all known canonical team names, sorted."""
        return sorted(self._reverse.keys())

    def get_all_aliases(self, canonical: str) -> list[str]:
        """Return all aliases for a canonical name."""
        return sorted(self._reverse.get(canonical, set()))

    # ── Internal ───────────────────────────────────────

    def _build_reverse_index(self) -> dict[str, set[str]]:
        """Build a canonical→{aliases} reverse index."""
        rev: dict[str, set[str]] = {}
        for alias, canonical in self._aliases.items():
            if canonical not in rev:
                rev[canonical] = set()
            rev[canonical].add(alias)
        return rev

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
