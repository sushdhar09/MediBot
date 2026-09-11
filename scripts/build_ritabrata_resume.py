"""Rebuild Ritabrata Dhar's AVP resume as a two-page DOCX with a premium side panel.

Source of truth: RitabrataDhar_HDFC Bank_AVP_new.pdf (revised version).
Every fact here comes from that document. Nothing is invented.
"""
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUTPUT = r"C:\code\MyJob_Applying_Agent\RitabrataDhar_HDFC_Bank_AVP_Enhanced.docx"

NAVY = "12324F"
GOLD_HEX = "C9A227"
BAND = "F2F5F9"
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xD5, 0xDF, 0xEA)
GOLD = RGBColor(0xC9, 0xA2, 0x27)
NAVY_RGB = RGBColor(0x12, 0x32, 0x4F)
DARK = RGBColor(0x23, 0x2B, 0x33)
GRAY = RGBColor(0x5A, 0x63, 0x6E)


def shade(cell, color):
    props = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), color)
    props.append(node)


def cell_margins(cell, top, start, bottom, end):
    props = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = OxmlElement(f"w:{name}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    props.append(tc_mar)


def no_borders(table):
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{name}")
        edge.set(qn("w:val"), "nil")
        borders.append(edge)
    table._tbl.tblPr.append(borders)


def underline(paragraph, color):
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "3")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    paragraph._p.get_or_add_pPr().append(pbdr)


def make_shell(doc):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(2.15)
    table.columns[1].width = Inches(5.35)
    no_borders(table)
    left, right = table.rows[0].cells
    left.width = Inches(2.15)
    right.width = Inches(5.35)
    left.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    right.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    shade(left, NAVY)
    shade(right, "FFFFFF")
    cell_margins(left, 160, 200, 160, 200)
    cell_margins(right, 80, 260, 80, 110)
    left._element.remove(left.paragraphs[0]._element)
    right._element.remove(right.paragraphs[0]._element)
    return left, right


def side_heading(cell, text, first=False):
    p = cell.add_paragraph()
    p.paragraph_format.space_before = Pt(2 if first else 11)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.name = "Aptos Display"
    r.font.size = Pt(9.2)
    r.font.color.rgb = GOLD
    underline(p, GOLD_HEX)


def side_line(cell, text, detail=None, bullet=False):
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"\u2022  {text}" if bullet else text)
    r.bold = detail is not None
    r.font.name = "Aptos"
    r.font.size = Pt(8.8)
    r.font.color.rgb = WHITE
    if detail:
        d = p.add_run(f"\n{detail}")
        d.font.name = "Aptos"
        d.font.size = Pt(8.5)
        d.font.color.rgb = PALE


def main_heading(cell, text):
    p = cell.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text.upper())
    r.bold = True
    r.font.name = "Aptos Display"
    r.font.size = Pt(11)
    r.font.color.rgb = NAVY_RGB
    underline(p, "9FB3C8")


def role(cell, title, company, dates, headline=None):
    p = cell.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    r.bold = True
    r.font.name = "Aptos"
    r.font.size = Pt(10.5)
    r.font.color.rgb = NAVY_RGB
    if company:
        r = p.add_run(f"   |   {company}")
        r.bold = True
        r.font.name = "Aptos"
        r.font.size = Pt(10)
        r.font.color.rgb = DARK

    d = cell.add_paragraph()
    d.paragraph_format.space_after = Pt(2)
    d.paragraph_format.keep_with_next = True
    r = d.add_run(dates)
    r.italic = True
    r.font.name = "Aptos"
    r.font.size = Pt(8.8)
    r.font.color.rgb = GRAY
    if headline:
        r = d.add_run(f"   \u2022   {headline}")
        r.bold = True
        r.font.name = "Aptos"
        r.font.size = Pt(8.8)
        r.font.color.rgb = GOLD


def bullet(cell, text):
    p = cell.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_together = True
    r = p.add_run("\u25aa  ")
    r.font.name = "Aptos"
    r.font.size = Pt(9.5)
    r.font.color.rgb = GOLD
    r = p.add_run(text)
    r.font.name = "Aptos"
    r.font.size = Pt(9.5)
    r.font.color.rgb = DARK


def metric_band(cell, metrics):
    table = cell.add_table(rows=1, cols=len(metrics))
    table.autofit = False
    no_borders(table)
    for index, (value, label) in enumerate(metrics):
        c = table.rows[0].cells[index]
        c.width = Inches(5.35 / len(metrics))
        shade(c, BAND)
        cell_margins(c, 60, 80, 60, 80)
        c._element.remove(c.paragraphs[0]._element)
        p = c.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(value)
        r.bold = True
        r.font.name = "Aptos Display"
        r.font.size = Pt(13)
        r.font.color.rgb = NAVY_RGB
        p = c.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(label.upper())
        r.bold = True
        r.font.name = "Aptos"
        r.font.size = Pt(7.6)
        r.font.color.rgb = GRAY


doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Aptos"
normal.font.size = Pt(10)
normal.font.color.rgb = DARK
normal.paragraph_format.space_after = Pt(3)

section = doc.sections[0]
section.top_margin = Inches(0.34)
section.bottom_margin = Inches(0.34)
section.left_margin = Inches(0.4)
section.right_margin = Inches(0.4)

# ------------------------------------------------------------------ page one
left, right = make_shell(doc)

p = left.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(1)
r = p.add_run("RD")
r.bold = True
r.font.name = "Aptos Display"
r.font.size = Pt(30)
r.font.color.rgb = GOLD

p = left.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
r = p.add_run("AVP & HEAD\nRELATIONSHIP BANKING")
r.bold = True
r.font.name = "Aptos"
r.font.size = Pt(9)
r.font.color.rgb = PALE

side_heading(left, "Contact", first=True)
side_line(left, "Mobile", "+91 99866 41475")
side_line(left, "Email", "ritabrata.dhar@gmail.com")
side_line(left, "LinkedIn", "linkedin.com/in/ritabrata-dhar-7790a520")
side_line(left, "Location", "Bengaluru, India")

side_heading(left, "Core Strengths")
for item in (
    "Retail & Relationship Banking",
    "HNI Portfolio Management",
    "Cross-Selling & Revenue Growth",
    "Branch Operations & P&L",
    "B2B Sales & Distribution",
    "Key Account Management",
    "Team Leadership & Coaching",
    "Risk, Compliance & Audit",
):
    side_line(left, item, bullet=True)

side_heading(left, "Banking & Product Depth")
for item in (
    "Cash Management Services",
    "Digital Payment Solutions",
    "API & Connected Banking",
    "Retail Liabilities (CASA, TD)",
    "Retail & Wholesale Assets",
    "Institutional Banking",
    "RTFx & Third-Party Products",
):
    side_line(left, item, bullet=True)

p = right.add_paragraph()
p.paragraph_format.space_after = Pt(0)
r = p.add_run("RITABRATA DHAR")
r.bold = True
r.font.name = "Aptos Display"
r.font.size = Pt(25)
r.font.color.rgb = NAVY_RGB

p = right.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("RETAIL BANKING LEADER  \u2022  RELATIONSHIP MANAGEMENT  \u2022  B2B SALES & DISTRIBUTION")
r.bold = True
r.font.name = "Aptos"
r.font.size = Pt(10)
r.font.color.rgb = GOLD

p = right.add_paragraph()
p.paragraph_format.space_after = Pt(7)
r = p.add_run("IIM Lucknow (MDP, Fintech, Banking & Risk Management)  |  MBA Marketing, ICFAI Business School Ahmedabad")
r.font.name = "Aptos"
r.font.size = Pt(9)
r.font.color.rgb = GRAY

metric_band(
    right,
    [
        ("\u20b91,800 Cr", "peak branch book"),
        ("\u20b9800 Cr", "HNI portfolio"),
        ("5", "RMs led"),
        ("16 yrs", "12 in banking"),
    ],
)

main_heading(right, "Executive Profile")
p = right.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.paragraph_format.line_spacing = 1.04
r = p.add_run(
    "Result-oriented retail banker with 16 years across retail and B2B sales and distribution, including "
    "12 years in banking. Delivers exceptional customer service, actively cross-sells banking products to "
    "meet sales goals, and builds strong, lasting client relationships. Skilled at identifying individual and "
    "corporate financial needs and tailoring solutions for optimal customer satisfaction. Excellent exposure "
    "to Cash Management Services, Digital Payment Solutions, API Banking and Connected Banking, with rich "
    "experience across Retail Liability, Retail Asset and Institutional Banking. Trusted with progressively "
    "larger books \u2014 from a \u20b990 crore branch to a \u20b91,800 crore liability branch and a "
    "\u20b9800 crore HNI relationship portfolio."
)
r.font.name = "Aptos"
r.font.size = Pt(9.5)
r.font.color.rgb = DARK

main_heading(right, "Career Highlights")

role(
    right,
    "AVP & Head Relationship Banking",
    "HDFC Bank Limited",
    "Apr 2022 \u2013 Present",
    "\u20b9800 Cr HNI liability book  |  5 Relationship Managers",
)
bullet(right, "Drive a team of five HNI Relationship Managers managing a \u20b9800 crore portfolio liability book against assigned targets in Liabilities (CASA and Term Deposits), Retail and Wholesale Assets, RTFx, Third-Party Products and Investments.")
bullet(right, "Ensure Income, Liability, Asset and Customer Engagement parameters are met while upholding the bank's compliance and hygiene parameters, with customer satisfaction paramount.")
bullet(right, "Build and maintain strong relationships with a diverse clientele, understanding their financial needs and providing tailored solutions.")
bullet(right, "Partner with stakeholders, including Relationship Managers and customers, to gather, assess and document business requirements for banking products and services.")
bullet(right, "Analyse client feedback and market trends to identify process improvement opportunities, enhancing customer experience and increasing revenue.")
bullet(right, "Conduct thorough data analysis to extract insights and drive data-driven decision-making, identifying cross-selling opportunities and targeted marketing strategies.")

role(
    right,
    "AVP & Branch Head",
    "Axis Bank Limited",
    "Oct 2021 \u2013 Apr 2022",
    "\u20b91,800 Cr liability book",
)
bullet(right, "Ensured efficient daily operation of a branch with a \u20b91,800 crore liability book \u2014 lending, product sales and customer service \u2014 in accordance with the bank's objectives.")
bullet(right, "Prepared the branch growth plan with the Cluster Head, implemented it through the branch team, and strengthened relations with key customers (top 10%) to generate sustained business.")
bullet(right, "Reviewed daily and periodic reports \u2014 overdrawn accounts, temporary overdrafts, cash retention limits \u2014 taking proactive action to ensure profitable and ethical business.")
bullet(right, "Met internal and external audit deliverables per prescribed norms and created a performance-oriented environment, ensuring all staff were trained on products, sales processes and policies.")

# ------------------------------------------------------------------ page two
# A full-size paragraph here would not fit under the page-one table and would
# spill, creating a blank page. Collapse it to a hairline so the break lands
# on page one and page two starts immediately.
breaker = doc.add_paragraph()
breaker.paragraph_format.space_before = Pt(0)
breaker.paragraph_format.space_after = Pt(0)
breaker.paragraph_format.line_spacing = Pt(1)
breaker_run = breaker.add_run()
breaker_run.font.size = Pt(1)
breaker_run.add_break(WD_BREAK.PAGE)

left2, right2 = make_shell(doc)

p = left2.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run("RITABRATA DHAR")
r.bold = True
r.font.name = "Aptos Display"
r.font.size = Pt(13)
r.font.color.rgb = WHITE

side_heading(left2, "Education", first=True)
side_line(left2, "MDP \u2014 Fintech, Banking & Risk Management", "IIM Lucknow, 2021")
side_line(left2, "MBA, Marketing", "ICFAI Business School Ahmedabad,\nICFAI University, 2008")
side_line(left2, "B.E., Electronics & Telecommunication", "C.V. Raman College of Engineering,\nUtkal University, 2004")
side_line(left2, "HSC \u2014 I.S.C. Board", "M.G.M English Medium School, 2000")
side_line(left2, "SSC \u2014 C.B.S.E. Board", "Kendriya Vidyalaya, 1998")

side_heading(left2, "Systems & Technology")
for item in (
    "Finacle (core banking)",
    "FLEXCUBE (core banking)",
    "Microsoft Office Suite",
    "SAP R/3 (working knowledge)",
    "C, C++",
):
    side_line(left2, item, bullet=True)

side_heading(left2, "Languages")
side_line(left2, "English  \u2022  Hindi", "Bengali  \u2022  Oriya")

side_heading(left2, "Personal")
side_line(left2, "Date of Birth", "5 February 1982")
side_line(left2, "Location", "Bengaluru 560068, India")
side_line(left2, "References", "Available on request")

role(
    right2,
    "Chief Manager & Branch Manager",
    "ICICI Bank Limited",
    "Aug 2013 \u2013 Sep 2021",
    "3 branches  |  \u20b990 Cr, \u20b9120 Cr, \u20b91,200 Cr books",
)
bullet(right2, "Executed and monitored overall administration and daily operations of full-service branch offices \u2014 operations, lending, product sales, customer service, security and safety \u2014 across three branches with liability books of \u20b990 crore, \u20b9120 crore and \u20b91,200 crore.")
bullet(right2, "Achieved incremental number and value targets for Liabilities (CA, SA, FD), Assets (Home, Auto and others) and Fee Products (Mutual Funds, Life and General Insurance, Gold).")
bullet(right2, "Prepared and monitored the branch sales plan and built a healthy asset and liability book through GL growth.")
bullet(right2, "Increased market share and managed key branch relationships.")
bullet(right2, "Achieved branch FOCUS and Customer Service Index score targets, holding wait time within permissible segmental limits with nil critical requests and nil escalations.")
bullet(right2, "Ensured operations, risk control and process adherence \u2014 branch and SOAX audit scores, fraud prevention, ops risk monitoring, RBI audits, inspections and incognito visits.")

role(right2, "Key Account Manager", "Orange Chemicals", "Feb 2013 \u2013 Aug 2013")
bullet(right2, "Sourced products against enquiries received from Principals and visited suppliers and distributors regularly.")
bullet(right2, "Arranged exclusive distribution tie-ups and provided techno-commercial back-up to the sourcing team.")
bullet(right2, "Followed up with principals for new enquiries and for receipt of commission and other payments while managing key accounts.")

role(right2, "Business Development Manager", "Pangea Chemicals Pvt. Ltd.", "Jul 2009 \u2013 Jan 2012")
bullet(right2, "Planned, forecast and prioritised key action areas across regions to achieve the set strategy, identifying and developing new markets and customers.")
bullet(right2, "Undertook regular business trips to market and maintain relationships with existing customers, serving clients and resolving queries.")
bullet(right2, "Coordinated with purchase, logistics and documentation to drive efficiency across the customer management process \u2014 quotations, commercial negotiation, delivery of goods and payment reception.")
bullet(right2, "Improved efficiency across the value chain by developing constructive solutions to overcome obstacles in sales and marketing activities.")

role(right2, "Marketing Manager", "Dey and Brothers", "Feb 2008 \u2013 Jul 2009")
bullet(right2, "Assessed market and current trends, identified competitors and recommended product improvements.")
bullet(right2, "Coordinated sales distribution by establishing sales territories, quotas and goals, and established training programmes for sales representatives.")
bullet(right2, "Analysed sales statistics to determine sales potential and inventory requirements and monitored customer preferences.")
bullet(right2, "Forecast marketing returns and expenses; managed alliance partners as main point of contact and created systems to streamline partner management.")

for sec in doc.sections:
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("Ritabrata Dhar  |  AVP & Head Relationship Banking  |  +91 99866 41475  |  ritabrata.dhar@gmail.com")
    r.font.name = "Aptos"
    r.font.size = Pt(7.5)
    r.font.color.rgb = GRAY

doc.core_properties.title = "Ritabrata Dhar - AVP & Head Relationship Banking"
doc.core_properties.author = "Ritabrata Dhar"
doc.core_properties.subject = "Retail banking, relationship management and B2B sales leadership"
doc.core_properties.keywords = (
    "retail banking, relationship banking, HNI, CASA, cash management, API banking, "
    "digital payments, Finacle, FLEXCUBE, branch banking, AVP"
)
doc.save(OUTPUT)
print(OUTPUT)
