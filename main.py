import os
import json
import re
import logging
import copy
import numpy as np
import cv2

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ['USERPROFILE'] = PROJECT_ROOT

LOCAL_POPPLER_PATH = os.path.join(PROJECT_ROOT, "poppler", "Library", "bin")
if os.path.exists(LOCAL_POPPLER_PATH):
    os.environ["PATH"] += os.pathsep + LOCAL_POPPLER_PATH

from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from schema_template import get_valuation_report_schema

logging.getLogger("ppocr").setLevel(logging.ERROR)

class ValuationReportParser:
    def __init__(self):
        print("Initializing Logic-Enhanced OCR Engine (Ultimate Complete Mode)...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        self.stop_markers = [
            "PROPERTY TYPE", "ACCOMMODATION", "BUILDING SURVEY", "VALUATION", 
            "Report Date", "CURRENT OCCUPANCY", "CONSTRUCTION", "LOCALITY & DEMAND", 
            "SERVICES", "ENERGY EFFICIENCY", "ESSENTIAL REPAIRS", "RENTAL INFORMATION", 
            "VALUATION FOR FINANCE", "GENERAL REMARKS", "VALUERS DECLARATION", 
            "IMPORTANT NOTICE", "See Continuation Page"
        ]

    def clean_text(self, text, keywords=None):
        if not text: return ""
        res = text
        patterns = [
            r"Applicant\(s\)\s*Surname\(s\)\s*&\s*Initials:?\.?",
            r"Application Number:?\.?", r"Date of Inspection:?\.?", r"Property Address:?\.?", r"Postcode:?\.?", 
            r"Full Name of Valuer:?\.?", r"Name of Valuer:?\.?", r"behalf of:?\.?", r"Telephone:?\.?", r"E-mail:?\.?"
        ]
        if keywords:
            for k in keywords: patterns.append(re.escape(k))
        for p in patterns:
            res = re.sub(p, "", res, flags=re.IGNORECASE).strip()
        return res.strip(": .-()&/")

    def get_elements(self, pdf_path):
        try:
            p_path = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None
            images = convert_from_path(pdf_path, poppler_path=p_path, dpi=300)
        except Exception as e:
            print(f"Error: {e}"); return []

        elements = []
        for p_idx, img in enumerate(images):
            img_np = np.array(img); img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            result = self.ocr.ocr(img_cv, cls=True)
            if not result or result[0] is None: continue
            for line in result[0]:
                coords = line[0]; text = line[1][0]
                xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
                elements.append({
                    "page": p_idx, "text": text.strip(),
                    "x1": min(xs), "x2": max(xs), "y1": min(ys), "y2": max(ys),
                    "cy": (min(ys) + max(ys)) / 2, "cx": (min(xs) + max(xs)) / 2
                })
        return elements

    def find_anchors(self, elements, keywords, page=None, exact=False):
        matches = []
        for el in elements:
            if page is not None and el["page"] != page: continue
            txt = el["text"].strip(": ").lower()
            if any(k.lower() == txt for k in keywords): matches.append(el)
        if not matches and not exact:
            for el in elements:
                if page is not None and el["page"] != page: continue
                if any(k.lower() in el["text"].lower() for k in keywords): matches.append(el)
        return matches

    def extract_text(self, elements, keywords, loc="right", page=None):
        anchors = self.find_anchors(elements, keywords, page)
        if not anchors: return None
        for anchor in anchors:
            inline = self.clean_text(anchor["text"], keywords)
            if len(inline) > 1: return inline
            candidates = []
            for el in elements:
                if el == anchor or el["page"] != anchor["page"]: continue
                if loc == "right":
                    if abs(el["cy"] - anchor["cy"]) < 40 and el["x1"] > anchor["x1"] and (el["x1"] - anchor["x2"]) < 600:
                        candidates.append((el["x1"]-anchor["x2"], el))
                elif loc == "bottom":
                    if el["cy"] > anchor["cy"] and (el["y1"] - anchor["y2"]) < 100 and abs(el["cx"] - anchor["cx"]) < 600:
                        candidates.append((el["y1"]-anchor["y2"], el))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                res = self.clean_text(candidates[0][1]["text"], keywords)
                if res and not any(s.lower() in res.lower()[:15] for s in self.stop_markers): return res
        return None

    def extract_int(self, elements, keywords, loc="right", page=None):
        anchors = self.find_anchors(elements, keywords, page)
        if not anchors: return None
        for anchor in anchors:
            inline = re.findall(r'\b\d+\b', anchor["text"])
            if inline: return int(inline[-1])
            candidates = []
            for el in elements:
                if el["page"] == anchor["page"] and el != anchor:
                    if abs(el["cy"] - anchor["cy"]) < 40 and el["x1"] > anchor["x1"] and (el["x1"] - anchor["x2"]) < 600:
                        candidates.append((el["x1"]-anchor["x2"], el))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                num = re.findall(r'\b\d+\b', candidates[0][1]["text"])
                if num: return int(num[0])
        return None

    def extract_bool(self, elements, keywords, loc="left", page=None):
        anchors = self.find_anchors(elements, keywords, page)
        if not anchors: return None
        for anchor in anchors:
            if re.search(r'(\b[xX☑v]\b)|(\[[xX☑v]\])|([xX☑v]$)|(Yes ?[xX☑v])', anchor["text"]): return True
            for el in elements:
                if el["page"] == anchor["page"] and el != anchor:
                    if abs(el["cy"] - anchor["cy"]) < 50 and abs(el["cx"] - anchor["cx"]) < 200:
                        if re.search(r'(\b[xX☑v]\b)|(\[[xX☑v]\])|([xX☑v]$)|(Yes ?[xX☑v])|ok', el["text"]): return True
        return False

    def extract_multiline(self, elements, keywords, page=None):
        anchors = self.find_anchors(elements, keywords, page)
        if not anchors: return None
        start = anchors[0]; lines = []
        for el in elements:
            if el["page"] == start["page"] and el["y1"] >= start["y1"] - 10 and el["y1"] - start["y2"] < 800:
                txt = self.clean_text(el["text"], keywords)
                if not txt: continue
                if el != start:
                    if any(txt.upper().startswith(s.upper()) for s in self.stop_markers if s != "Postcode"): break
                if abs(el["x1"] - start["x1"]) < 1200 or abs(el["cx"] - start["cx"]) < 600:
                    lines.append(txt)
                    if txt.upper().startswith("POSTCODE") and el != start: break
        return " ".join(dict.fromkeys(lines)) if lines else None

    def extract_currency(self, elements, keywords, page=None):
        anchors = self.find_anchors(elements, keywords, page, exact=True)
        if not anchors: return None
        for anchor in anchors:
            candidates = []
            for el in elements:
                if el["page"] == anchor["page"] and el != anchor:
                    # ULTRA-STRICT Y match
                    if abs(el["cy"] - anchor["cy"]) < 15 and el["x1"] > anchor["x2"] and (el["x1"] - anchor["x2"]) < 600:
                        candidates.append((abs(el["x1"]-anchor["x2"]), el))
                    elif el["y1"] > anchor["y2"] and (el["y1"] - anchor["y2"]) < 100 and abs(el["cx"] - anchor["cx"]) < 400:
                        candidates.append((abs(el["y1"]-anchor["y2"]), el))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                for _, el in candidates:
                    v_str = re.sub(r'[^0-9.]', '', el["text"].replace(",", ""))
                    if v_str and v_str.replace('.', '', 1).isdigit(): return float(v_str)
        return None

    def parse(self, pdf_path):
        print(f"Parsing: {os.path.basename(pdf_path)}")
        els = self.get_elements(pdf_path); d = get_valuation_report_schema()
        d["applicationNumber"] = self.extract_text(els, ["Application Number"], page=0)
        if d["applicationNumber"]: d["applicationNumber"] = re.sub(r"\s+", "", d["applicationNumber"])
        d["applicantName"] = self.extract_text(els, ["Surname", "Initials"], "bottom", page=0)
        d["dateOfInspection"] = self.extract_text(els, ["Date of Inspection"], page=0)
        d["propertyAddress"] = self.extract_multiline(els, ["Property Address"], page=0)
        addr = d["propertyAddress"] or ""
        pc = re.search(r'\b[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}\b', addr)
        d["postCode"] = self.extract_text(els, ["Postcode"], "right", page=0) or (pc.group(0) if pc else None)
        pt = d["propertyType"]
        for k, l in [("isDetachedHouse","Detached House"),("isSemiDetachedHouse","Semi-Detached"),("isTerracedHouse","Terraced House"),("isBungalow","Bungalow"),("isFlat","Flat"),("isMaisonette","Maisonette")]:
            pt[k] = self.extract_bool(els, [l])
        pt["tenure"] = self.extract_text(els, ["Tenure"], "right")
        pt["remainingLeaseTermYears"] = self.extract_int(els, ["Remaining term of Lease"])
        vf = d["valuationForFinancePurpose"]
        vf["marketValuePresentCondition"] = self.extract_currency(els, ["Market Value in present condition"])
        vf["marketValueAfterRepairs"] = self.extract_currency(els, ["Market Value after essential repairs/completion"])
        vf["buildingInsuranceReinstatementCost"] = self.extract_currency(els, ["Building Insurance Reinstatement Cost"])
        d["valuationForFinancePurposeHPP"] = copy.deepcopy(vf)
        d["generalRemarks"] = self.extract_multiline(els, ["GENERAL REMARKS"])
        vd = d["valuersDeclaration"]
        vd["valuerName"] = self.extract_text(els, ["Name of Valuer"])
        vd["email"] = self.extract_text(els, ["E-mail"])
        vd["ricsNumber"] = self.extract_int(els, ["RICS Number"])
        return d, els

if __name__ == "__main__":
    parser = ValuationReportParser()
    files = ["Ashok.test Valuation Report.pdf", "All yes fields (1).pdf", "All empty (1).pdf"]
    output = {}; raw_data = {}
    for f in files:
        if os.path.exists(f):
            key = f.split('.')[0].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower() + "_report"
            res, els = parser.parse(f); output[key] = res; raw_data[key] = els
    with open("extracted_data.json", "w") as f: json.dump(output, f, indent=4)
    with open("raw_ocr_data.json", "w") as f: json.dump(raw_data, f, indent=4)
    print("Optimization Complete.")
