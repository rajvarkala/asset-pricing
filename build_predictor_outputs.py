from pathlib import Path
import re

import fitz
import pandas as pd
import pdfplumber

root = Path('/Users/raj/ws/quantconnect')
ml_pdf = root / 'ML.pdf'
supp_pdf = root / 'hhaa009_supplementary_data.pdf'
out_csv = root / 'top_predictors_page33_dedup_enriched.csv'
out_xlsx = root / 'top_predictors_page33_dedup_enriched.xlsx'


def normalize_acronym(s: str) -> str:
    s = s.strip().lower()
    s = s.replace('\ufb01', 'fi').replace('\ufb00', 'ff')
    s = re.sub(r'\s+', '_', s)
    return s


# Parse Table A.6 rows (94 characteristics) from supplementary PDF pages 14-16.
doc = fitz.open(str(supp_pdf))
lines = []
for pg in [14, 15, 16]:
    lines.extend(doc[pg - 1].get_text('text').splitlines())

a6_map = {}
for i, ln in enumerate(lines):
    cur = ln.strip()
    if re.fullmatch(r'\d{1,2}', cur):
        num = int(cur)
        if 1 <= num <= 94 and i + 2 < len(lines):
            acr = normalize_acronym(lines[i + 1])
            desc = lines[i + 2].strip()
            a6_map.setdefault(acr, desc)

a6_map.setdefault('sic2', 'Industry dummy based on first two digits of SIC code')

# Extract deduplicated top variables from Figure 4 on page 33 in main PDF.
with pdfplumber.open(str(ml_pdf)) as pdf:
    p33 = pdf.pages[32].extract_text() or ''

all_codes = set(a6_map.keys())
tokens = [normalize_acronym(t) for t in re.findall(r'[A-Za-z0-9_]+', p33)]
figure4_codes = [t for t in tokens if t in all_codes]

seen = set()
ordered = []
for c in figure4_codes:
    if c not in seen:
        seen.add(c)
        ordered.append(c)

freq = pd.Series(figure4_codes).value_counts().to_dict()


def category(code: str) -> str:
    momentum = {'mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom', 'indmom', 'maxret'}
    liquidity = {'turn', 'std_turn', 'mvel1', 'dolvol', 'ill', 'zerotrade', 'baspread', 'std_dolvol'}
    risk = {'retvol', 'idiovol', 'beta', 'betasq'}
    valuation_fundamental = {
        'ep', 'sp', 'agr', 'nincr', 'chcsho', 'cashpr', 'invest', 'lgr', 'bm', 'bm_ia', 'operprof', 'roaq', 'roeq', 'roic'
    }
    industry = {'sic2', 'securedind', 'convind'}
    if code in momentum:
        return 'Price trend / momentum'
    if code in liquidity:
        return 'Liquidity / trading activity'
    if code in risk:
        return 'Risk exposure / volatility'
    if code in valuation_fundamental:
        return 'Valuation / fundamentals'
    if code in industry:
        return 'Industry / structure'
    return 'Other characteristic'


rows = []
for rank, code in enumerate(ordered, start=1):
    rows.append(
        {
            'rank_first_appearance_page33': rank,
            'predictor_variable': code,
            'appearance_count_in_figure4': int(freq.get(code, 0)),
            'firm_characteristic_description': a6_map.get(code, ''),
            'grouping': category(code),
            'paper_source': 'ML.pdf Figure 4 (page 33)',
            'definition_source': 'hhaa009_supplementary_data.pdf Table A.6',
            'online_enrichment_note': 'CRSP/Compustat variables are WRDS-hosted datasets; SIC grouping follows U.S. SIC major-group structure.',
        }
    )

top_df = pd.DataFrame(rows)

# Macroeconomic predictors from Section 2.1, page 26 of ML.pdf
macro_rows = [
    ('dp', 'Dividend-price ratio', 'Welch & Goyal (2008) predictor set'),
    ('ep', 'Earnings-price ratio', 'Welch & Goyal (2008) predictor set'),
    ('bm', 'Book-to-market ratio', 'Welch & Goyal (2008) predictor set'),
    ('ntis', 'Net equity expansion', 'Welch & Goyal (2008) predictor set'),
    ('tbl', 'Treasury-bill rate', 'Welch & Goyal (2008) predictor set'),
    ('tms', 'Term spread', 'Welch & Goyal (2008) predictor set'),
    ('dfy', 'Default spread', 'Welch & Goyal (2008) predictor set'),
    ('svar', 'Stock variance', 'Welch & Goyal (2008) predictor set'),
]
macro_df = pd.DataFrame(macro_rows, columns=['indicator', 'description', 'source'])
macro_df['paper_source'] = 'ML.pdf page 26, Section 2.1'
macro_df['online_enrichment_note'] = (
    'Predictor definitions follow Welch & Goyal (2008); monthly data referenced from Amit Goyal data library in paper footnote.'
)

# SIC codes sheet: inferred 74 first-two-digit major groups used as dummies.
sic_groups = [
    ('01', 'Agricultural Production Crops', 'A'), ('02', 'Agriculture Production Livestock And Animal Specialties', 'A'),
    ('07', 'Agricultural Services', 'A'), ('08', 'Forestry', 'A'), ('09', 'Fishing, Hunting, And Trapping', 'A'),
    ('10', 'Metal Mining', 'B'), ('12', 'Coal Mining', 'B'), ('13', 'Oil And Gas Extraction', 'B'),
    ('14', 'Mining And Quarrying Of Nonmetallic Minerals, Except Fuels', 'B'), ('15', 'Building Construction General Contractors And Operative Builders', 'C'),
    ('16', 'Heavy Construction Other Than Building Construction Contractors', 'C'), ('17', 'Construction Special Trade Contractors', 'C'),
    ('20', 'Food And Kindred Products', 'D'), ('21', 'Tobacco Products', 'D'), ('22', 'Textile Mill Products', 'D'),
    ('23', 'Apparel And Other Finished Products Made From Fabrics And Similar Materials', 'D'), ('24', 'Lumber And Wood Products, Except Furniture', 'D'),
    ('25', 'Furniture And Fixtures', 'D'), ('26', 'Paper And Allied Products', 'D'), ('27', 'Printing, Publishing, And Allied Industries', 'D'),
    ('28', 'Chemicals And Allied Products', 'D'), ('29', 'Petroleum Refining And Related Industries', 'D'),
    ('30', 'Rubber And Miscellaneous Plastics Products', 'D'), ('31', 'Leather And Leather Products', 'D'),
    ('32', 'Stone, Clay, Glass, And Concrete Products', 'D'), ('33', 'Primary Metal Industries', 'D'),
    ('34', 'Fabricated Metal Products, Except Machinery And Transportation Equipment', 'D'), ('35', 'Industrial And Commercial Machinery And Computer Equipment', 'D'),
    ('36', 'Electronic And Other Electrical Equipment And Components, Except Computer Equipment', 'D'), ('37', 'Transportation Equipment', 'D'),
    ('38', 'Measuring, Analyzing, And Controlling Instruments; Photographic, Medical And Optical Goods; Watches And Clocks', 'D'), ('39', 'Miscellaneous Manufacturing Industries', 'D'),
    ('40', 'Railroad Transportation', 'E'), ('41', 'Local And Suburban Transit And Interurban Highway Passenger Transportation', 'E'),
    ('42', 'Motor Freight Transportation And Warehousing', 'E'), ('43', 'United States Postal Service', 'E'), ('44', 'Water Transportation', 'E'),
    ('45', 'Transportation By Air', 'E'), ('46', 'Pipelines, Except Natural Gas', 'E'), ('47', 'Transportation Services', 'E'),
    ('48', 'Communications', 'E'), ('49', 'Electric, Gas, And Sanitary Services', 'E'), ('50', 'Wholesale Trade-durable Goods', 'F'),
    ('51', 'Wholesale Trade-non-durable Goods', 'F'), ('52', 'Building Materials, Hardware, Garden Supply, And Mobile Home Dealers', 'G'),
    ('53', 'General Merchandise Stores', 'G'), ('54', 'Food Stores', 'G'), ('55', 'Automotive Dealers And Gasoline Service Stations', 'G'),
    ('56', 'Apparel And Accessory Stores', 'G'), ('57', 'Home Furniture, Furnishings, And Equipment Stores', 'G'),
    ('58', 'Eating And Drinking Places', 'G'), ('59', 'Miscellaneous Retail', 'G'), ('60', 'Depository Institutions', 'H'),
    ('61', 'Non-depository Credit Institutions', 'H'), ('62', 'Security And Commodity Brokers, Dealers, Exchanges, And Services', 'H'),
    ('63', 'Insurance Carriers', 'H'), ('64', 'Insurance Agents, Brokers, And Service', 'H'), ('65', 'Real Estate', 'H'),
    ('67', 'Holding And Other Investment Offices', 'H'), ('70', 'Hotels, Rooming Houses, Camps, And Other Lodging Places', 'I'),
    ('72', 'Personal Services', 'I'), ('73', 'Business Services', 'I'), ('75', 'Automotive Repair, Services, And Parking', 'I'),
    ('76', 'Miscellaneous Repair Services', 'I'), ('78', 'Motion Pictures', 'I'), ('79', 'Amusement And Recreation Services', 'I'),
    ('80', 'Health Services', 'I'), ('81', 'Legal Services', 'I'), ('82', 'Educational Services', 'I'),
    ('83', 'Social Services', 'I'), ('84', 'Museums, Art Galleries, And Botanical And Zoological Gardens', 'I'),
    ('86', 'Membership Organizations', 'I'), ('87', 'Engineering, Accounting, Research, Management, And Related Services', 'I'),
    ('89', 'Miscellaneous Services', 'I'),
]

sic_df = pd.DataFrame(sic_groups, columns=['sic2_code', 'major_group_name', 'division'])
sic_df['paper_source'] = 'ML.pdf page 26: 74 industry dummies using first two-digit SIC'
sic_df['online_reference'] = 'https://www.osha.gov/data/sic-manual'
sic_df['construction_note'] = 'Inferred 74-group list aligned to paper count; based on SIC major groups typically represented in listed equities.'

top_df.to_csv(out_csv, index=False)
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    top_df.to_excel(writer, index=False, sheet_name='TopPredictors_Page33')
    macro_df.to_excel(writer, index=False, sheet_name='MacroIndicators_8')
    sic_df.to_excel(writer, index=False, sheet_name='SIC2_Codes_74')

print('top_count', len(top_df))
print('macro_count', len(macro_df))
print('sic_count', len(sic_df))
print('csv', out_csv)
print('xlsx', out_xlsx)
print('top10', ', '.join(top_df['predictor_variable'].head(10).tolist()))
