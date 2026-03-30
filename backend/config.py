import os
from pathlib import Path

# Database
DB_URL = "postgresql://postgres:admin@localhost:5432/nepsegpt"

# ML Config
MODEL_DIR = str(Path(__file__).parent / "models_ml")
LOOKFORWARD_DAYS = 5
MIN_TRAIN_ROWS = 100
SIGNAL_THRESHOLD = 0.65
BUY_RETURN_PCT = 2.5
SELL_RETURN_PCT = -2.5

# Indicator Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 20
EMA_LONG = 50
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
STOCH_K = 14
STOCH_D = 3
WILLIAMS_R = 14
CCI_PERIOD = 20
MFI_PERIOD = 14

# Supabase (Optional/Placeholder for compatibility with provided code)

SUPABASE_URL = "" 
SUPABASE_KEY = ""

# NEPSE Sectors
SECTORS = {
    "Banking": ["NABIL", "ADBL", "BOKL", "CCBL", "CZBIL", "EBL", "GBIME", "KBL", "LBL", "MBL", "MEGA", "NBL", "NBBL", "NMB", "PCBL", "PRVU", "SBL", "SBI", "SCB", "SRBL"],
    "Development Bank": ["ADBL", "Bfc", "EDBL", "GBBL", "GRDBL", "JBBL", "KRBL", "KSBBL", "LBBL", "MDB", "MLBL", "MNBBL", "SADBL", "SAPDBL", "SHINE", "SINDU"],
    "Hotel & Tourism": ["OHL", "SHL", "TRH"],
    "Hydro Power": ["AKJCL", "AKPL", "API", "BARUN", "BPCL", "CHCL", "CHL", "GHL", "GLICL", "HDPC", "HPPL", "HURJA", "JOSHI", "KPCL", "LEC", "MEN", "NGPL", "NHDL", "NHPC", "PMHPL", "PPCL", "RADHI", "RHPC", "RHPL", "RRHP", "SHEL", "SHPC", "SJCL", "SPDL", "SSHL", "UMHL", "UMRH", "UNHPL", "UPCL", "UPPER"],
    "Life Insurance": ["ALICL", "GLICL", "ILICL", "LICN", "NLIC", "NLICL", "PLIC", "RLI", "SLICL", "ULICL"],
    "Microfinance": ["CBBL", "DDBL", "FOWAD", "GBLBS", "GILB", "GLBSL", "GMFBS", "ILBS", "JBLB", "KMCDB", "LLBS", "MERO", "MLBBL", "MSLB", "NESDO", "NIDL", "NMBMF", "NUBL", "RMDC", "RSDC", "SABSL", "SDLBSL", "SKBBL", "SLBSL", "SLBBL", "SMBB", "SMFDB", "SMP", "SWBBL", "USLB", "VLBS", "WNLB"],
    "Non Life Insurance": ["AIC", "EIC", "GIC", "HGI", "IGI", "LGIL", "NIL", "NICL", "NLG", "PRIN", "PIC", "PICL", "RBCL", "SGI", "SIC", "SICL", "UIC"],
    "Manufacturing & Processing": ["BNL", "HDL", "STC", "UNL"],
    "Others": ["CIT", "HIDCL", "NTC", "NRN"],
    "Investment": ["CHDC", "ENL", "GUFL", "ICFC", "MFIL", "MPFL", "Nfs", "RLFL", "SFCL", "SIFC"],
    "Trading": ["BBC", "STC"]
}
