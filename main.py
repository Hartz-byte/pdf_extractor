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
        print("Initializing Advanced-Reliability OCR Engine...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        self.stop_markers = [
            "PROPERTY TYPE", "ACCOMMODATION", "BUILDING SURVEY", "VALUATION", 
            "Report Date", "CURRENT OCCUPANCY", "CONSTRUCTION", "LOCALITY & DEMAND", 
            "SERVICES", "ENERGY EFFICIENCY", "ESSENTIAL REPAIRS", "RENTAL INFORMATION", 
            "VALUATION FOR FINANCE", "GENERAL REMARKS", "VALUERS DECLARATION", 
            "IMPORTANT NOTICE"
        ]
        
        self.boilerplate = [
            "behalf of any group", "generality of the foregoing", "mortgage administrator",
            "trustee on behalf", "interested in the mortgage", "opinion likely",
            "Gatehouse Bank", "please provide details"
        ]

    def clean_text(self, text, keywords=None):
        if not text: return ""
        res = text
        patterns = [
            r"Applicant\(s\)\s*Surname\(s\)\s*&\s*Initials:?\.?",
            r"Application Number:?\.?", r"Date of Inspection:?\.?", r"Property Address:?\.?", r"Postcode:?\.?", 
            r"Full Name of Valuer:?\.?", r"Name of Valuer:?\.?", r"Telephone:?\.?", r"E-mail:?\.?",
            r"approximate % of [a-z ]+", r"If (Yes|No),? [a-z ]+ details:?",
            r"For and (?=on behalf of)", r"Address of Valuer:?\.?", r"RICS Number:?\.?", r"Fax:?\.?",
            r"Signature of Valuer", r"electronic signature", r":$"
        ]
        if keywords:
            for k in keywords: patterns.append(re.escape(k) + r"[:. ]*")
        
        for p in patterns:
            res = re.sub(p, "", res, flags=re.IGNORECASE).strip()
        
        # Remove trailing boolean markers
        res = re.sub(r"[xX☑v]$|(?<=[a-z])[xX☑v]$|(?<=[a-z])[xX☑v](?=[A-Z])", "", res).strip()
        return res.strip(": .-()&/")

    def get_elements(self, pdf_path):
        try:
            p_path = LOCAL_POPPLER_PATH if os.path.exists(LOCAL_POPPLER_PATH) else None
            images = convert_from_path(pdf_path, poppler_path=p_path, dpi=300)
        except Exception as e:
            print(f"Error loading PDF: {e}"); return []

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

    def find_anchors(self, elements, keywords, page=None):
        matches = []
        for el in elements:
            if page is not None and el["page"] != page: continue
            if any(k.lower() in el["text"].lower() for k in keywords):
                matches.append(el)
        return matches

    def extract_text(self, elements, keywords, loc="right", page=None, max_dist=1200, max_len=None):
        anchors = self.find_anchors(elements, keywords, page)
        if not anchors: return None
        anchors.sort(key=lambda x: (x["page"], x["y1"]))
        
        for anchor in anchors:
            inline = self.clean_text(anchor["text"], keywords)
            if len(inline) > 2 and not any(b.lower() in inline.lower() for b in self.boilerplate): 
                if not any(s.lower() == inline.lower() for s in self.stop_markers):
                    if not (max_len and len(inline) > max_len): return inline 
            
            candidates = []
            for el in elements:
                if el == anchor or el["page"] != anchor["page"]: continue
                if loc == "right":
                    if abs(el["cy"] - anchor["cy"]) < 75 and el["x1"] > anchor["x1"] and (el["x1"] - anchor["x2"]) < max_dist:
                        candidates.append((el["x1"]-anchor["x2"], el))
                elif loc == "bottom":
                    if el["cy"] > anchor["cy"] and (el["y1"] - anchor["y2"]) < 180 and abs(el["cx"] - anchor["cx"]) < 650:
                        candidates.append((el["y1"]-anchor["y2"], el))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                for _, cand_el in candidates:
                    res = self.clean_text(cand_el["text"], keywords)
                    if res and len(res) > 1:
                        if not any(s.lower() == res.lower() for s in self.stop_markers): 
                            if not any(b.lower() in res.lower() for b in self.boilerplate):
                                if not (max_len and len(res) > max_len): return res
        return None

    def extract_int(self, elements, keywords, is_year=False, is_percent=False):
        anchors = self.find_anchors(elements, keywords)
        if not anchors: return None
        for anchor in anchors:
            # Inline numeric check
            num_in_anchor = re.search(r'(\d+)$', anchor["text"])
            if num_in_anchor:
                val = int(num_in_anchor.group(1))
                if is_percent and val > 100: val = int(str(val)[:2])
                if is_year and (val < 1800 or val > 2100): pass
                else: return val
            
            # Nearest neighbor check
            for el in elements:
                if el["page"] == anchor["page"] and el != anchor:
                    if abs(el["cy"] - anchor["cy"]) < 80 and (abs(el["x1"] - anchor["x2"]) < 950 or abs(el["cx"] - anchor["cx"]) < 450):
                        nums = re.findall(r'\b\d+\b', el["text"])
                        if nums:
                            val = int(nums[0])
                            if is_percent and val > 100: val = int(str(val)[:2])
                            if is_year and (val < 1800 or val > 2100): continue
                            return val
        return None

    def extract_bool(self, elements, keywords):
        anchors = self.find_anchors(elements, keywords)
        if not anchors: return None
        for anchor in anchors:
            text = anchor["text"]
            for k in keywords:
                k_pos = text.lower().find(k.lower())
                if k_pos != -1:
                    fragment = text[k_pos:k_pos+len(k)+25]
                    # Direct check in same element
                    if re.search(r'[xX☑v] (Yes|No|$)', fragment, re.I) or re.search(r'[xX☑v]$', fragment):
                        # Special check for "XNo" -> False
                        if re.search(r'[xX☑v]\s?No', fragment, re.I): return False
                        return True
                    # Check if 'Yes' followed by mark
                    if re.search(r'Yes\s?[xX☑v]', fragment, re.I): return True
            
            # Spatial search for marks
            for el in elements:
                if el["page"] == anchor["page"] and el != anchor:
                    if abs(el["cy"] - anchor["cy"]) < 100 and abs(el["cx"] - anchor["cx"]) < 450:
                        txt = el["text"].strip().lower()
                        if re.fullmatch(r'[xX☑v]', txt): return True
                        if "yes" in txt and any(m in txt for m in ["x", "v", "☑"]): return True
        return False

    def extract_multiline(self, elements, keywords, page=None, max_lines=6):
        start_anchors = self.find_anchors(elements, keywords, page)
        if not start_anchors: return None
        start = start_anchors[0]; lines = []
        
        inline = self.clean_text(start["text"], keywords)
        if len(inline) > 3: lines.append(inline)

        block_els = []
        for el in elements:
            if el["page"] == start["page"] and el != start:
                if el["y1"] >= start["y1"] - 20 and el["y1"] - start["y2"] < 600:
                    if abs(el["cx"] - start["cx"]) < 950:
                        block_els.append(el)
        
        block_els.sort(key=lambda x: (x["y1"], x["x1"]))
        for el in block_els:
            txt = self.clean_text(el["text"], keywords)
            if not txt or any(s.lower() == txt.lower() for s in self.stop_markers): continue
            if any(b.lower() in txt.lower() for b in self.boilerplate): continue
            lines.append(txt)
            if len(lines) >= max_lines: break
            
        final = " ".join(dict.fromkeys(lines))
        return final if len(final) > 3 else None

    def extract_currency(self, elements, keywords):
        anchors = self.find_anchors(elements, keywords)
        if not anchors: return None
        for anchor in anchors:
            curr_match = re.search(r'[£$]?\s?(\d[\d,.]+)', anchor["text"])
            if curr_match:
                try: return float(curr_match.group(1).replace(",", ""))
                except: pass
            
            for el in elements:
                if el["page"] == anchor["page"] and el != anchor:
                    if abs(el["cy"] - anchor["cy"]) < 80 and el["x1"] > anchor["x1"] and (el["x1"] - anchor["x2"]) < 1000:
                        v_str = "".join(re.findall(r'[0-9.,]{2,}', el["text"])).replace(",", "")
                        try: return float(v_str) if v_str else None
                        except: pass
        return None

    def parse(self, pdf_path):
        print(f"Logic-Processing: {os.path.basename(pdf_path)}")
        els = self.get_elements(pdf_path); d = get_valuation_report_schema()
        
        # 1. TOP LEVEL
        d["applicationType"] = self.extract_text(els, ["VALUATION REPORT"], page=0)
        d["applicationNumber"] = self.extract_text(els, ["Application Number"], page=0)
        if d["applicationNumber"]: d["applicationNumber"] = re.sub(r"\D", "", d["applicationNumber"])
        d["applicantName"] = self.extract_text(els, ["Surname", "Initials"], "bottom", page=0)
        d["dateOfInspection"] = self.extract_text(els, ["Date of Inspection"], page=0)
        d["propertyAddress"] = self.extract_multiline(els, ["Property Address"], page=0)
        d["postCode"] = self.extract_text(els, ["Postcode"])
        
        # 2. PROPERTY TYPE
        pt = d["propertyType"]
        for k, l in [("isDetachedHouse", ["Detached House"]), ("isSemiDetachedHouse", ["Semi-Detached"]),
                     ("isTerracedHouse", ["Terraced House"]), ("isBungalow", ["Bungalow"]),
                     ("isFlat", ["Flat"]), ("isMaisonette", ["Maisonette"]), ("isBuiltOrOwnedByLocalAuthority", ["Local Authority"]),
                     ("isFlatMaisonetteConverted", ["Converted"]), ("isPurposeBuilt", ["Purpose Built"]), 
                     ("isAboveCommercial", ["Above commercial"]), ("isFlyingFreehold", ["Flying freehold"]),
                     ("isPartCommercialUse", ["commercial use"]), ("isPurchasedUnderSharedOwnership", ["Shared Ownership"])]:
            pt[k] = self.extract_bool(els, l)
        
        pt["flatMaisonetteFloor"] = self.extract_int(els, ["what floor"])
        pt["numberOfFloorsInBlock"] = self.extract_int(els, ["floors in block"])
        pt["ownerOccupationPercentage"] = self.extract_int(els, ["owner occupation"], is_percent=True)
        pt["conversionYear"] = self.extract_int(els, ["year of conversion"], is_year=True)
        pt["numberOfUnitsInBlock"] = self.extract_int(els, ["units in block"])
        pt["residentialNatureImpact"] = self.extract_text(els, ["Noise", "Odour"], "bottom")
        pt["tenure"] = self.extract_text(els, ["Tenure"])
        pt["flyingFreeholdPercentage"] = self.extract_int(els, ["flying freehold", "percentage"], is_percent=True)
        pt["maintenanceCharge"] = self.extract_currency(els, ["Maintenance Charge"])
        pt["roadCharges"] = self.extract_currency(els, ["Road Charge", "Road Charge:"])
        pt["groundRent"] = self.extract_currency(els, ["Ground Rent", "Renu'r"]) # misread Ground Rent
        pt["remainingLeaseTermYears"] = self.extract_int(els, ["Remaining term of Lease"])
        pt["commercialUsePercentage"] = self.extract_int(els, ["commercial use", "percentage"], is_percent=True)
        pt["yearBuilt"] = self.extract_int(els, ["Year property built"], is_year=True)

        # 3. ACCOMMODATION
        acc = d["accommodation"]
        for k, l in [("hall", ["Hall"]), ("livingRooms", ["Living Rooms"]), ("kitchen", ["Kitchen"]), ("utility", ["Utility"]), 
                     ("bedrooms", ["Bedrooms"]), ("bathrooms", ["Bathrooms"]), ("separateWc", ["Separate WC"]), 
                     ("basement", ["Basement"]), ("garage", ["Garage"]), ("parking", ["Parking"])]:
            acc[k] = self.extract_int(els, l)
        acc["isLiftPresent"] = self.extract_bool(els, ["Lift"])
        acc["gardens"] = self.extract_bool(els, ["Gardens"])
        acc["isPrivate"] = self.extract_bool(els, ["Private"])
        acc["isCommunal"] = self.extract_bool(els, ["Communal"])
        acc["numberOfOutbuildings"] = self.extract_int(els, ["outbuildings"])
        acc["outbuildingDetails"] = self.extract_text(els, ["outbuilding details"])
        acc["grossFloorAreaOfDwelling"] = self.extract_int(els, ["Gross floor area"])

        # 4. OCCUPANCY
        occ = d["currentOccupency"]
        occ["isEverOccupied"] = self.extract_bool(els, ["ever been occupied"])
        occ["numberOfAdultsInProperty"] = self.extract_int(els, ["adults appear to live"])
        occ["isHmoOrMultiUnitFreeholdBlock"] = self.extract_bool(els, ["HMO/Multi Unit"])
        occ["isCurrentlyTenanted"] = self.extract_bool(els, ["tenanted at present"])
        occ["hmoOrMultiUnitDetails"] = self.extract_text(els, ["HMO details", "Multi Unit details"])

        # 5. NEW BUILD
        nb = d["newBuild"]
        for k, l in [("isNewBuildOrRecentlyConverted", ["New Build"]), ("isCompleted", ["Completed"]),
                     ("isUnderConstruction", ["Under Construction"]), ("isFinalInspectionRequired", ["Final inspection required"]),
                     ("isNhbcCert", ["NHBC"]), ("isBuildZone", ["Buildzone"]), ("isPremier", ["Premier"]),
                     ("isProfessionalConsultant", ["Professional Consultant"]), ("isOtherCert", ["Other Cert"]),
                     ("isSelfBuildProject", ["Self-build"]), ("isInvolvesPartExchange", ["part exchange"]),
                     ("isDisclosureOfIncentivesSeen", ["Disclosure of Incentives"])]:
            nb[k] = self.extract_bool(els, l)
        nb["otherCertDetails"] = self.extract_text(els, ["Other X", "provide details"])
        nb["incentivesDetails"] = self.extract_text(els, ["Including total value of incentives"])
        nb["newBuildDeveloperName"] = self.extract_text(els, ["Developer"])

        # 6. CONSTRUCTION & LOCALITY & SERVICES
        for section, mapping in [
            (d["construction"], [("isStandardConstruction", ["Standard construction"]), ("nonStandardConstructionType", ["system or type"]),
                                 ("isHasAlterationsOrExtensions", ["alterations", "extensions"]), ("isAlterationsRequireConsents", ["require consents"])]),
            (d["localityAndDemand"], [("isUrban", ["Urban"]), ("isSuburban", ["Suburban"]), ("isRural", ["Rural"]),
                                      ("isGoodMarketAppeal", ["Good"]), ("isAverageMarketAppeal", ["Average"]), ("isPoorMarketAppeal", ["Poor"]),
                                      ("isOwnerResidential", ["Owner residential"]), ("isResidentialLet", ["Residential let"]), ("isCommercial", ["Commercial"]),
                                      ("isPricesRising", ["Prices", "Rising"]), ("isPricesStatic", ["Prices", "Static"]), ("isPricesFalling", ["Prices", "Falling"]),
                                      ("isDemandRising", ["Demand", "Rising"]), ("isDemandStatic", ["Demand", "Static"]), ("isDemandFalling", ["Demand", "Falling"]),
                                      ("isAffectedByCompulsoryPurchase", ["Compulsory Purchase"]), ("isVacantOrBoardedPropertiesNearby", ["vacant or boarded"]),
                                      ("isOccupancyRestrictionPossible", ["Occupancy restriction"]), ("isCloseToHighVoltageEquipment", ["high voltage equipment"])]),
            (d["services"], [("isMainsWater", ["Mains"]), ("isPrivateWater", ["Private"]), ("isUnknownWater", ["Unknown"]),
                             ("isGasSupply", ["Gas"]), ("isElectricitySupply", ["Electricity"]), ("isCentralHeating", ["Central Heating"]),
                             ("isMainDrainage", ["Main drainage"]), ("isSepticTankPlant", ["Septic tank"]), ("isUnknownDrainage", ["Unknown drainage"]),
                             ("isSolarPanels", ["Solar panels"]), ("isSharedAccess", ["Shared access"]), ("isRoadAdopted", ["Road adopted"]),
                             ("isHasEasementsOrRightsOfWay", ["Easements", "Rights of Way"])])
        ]:
            for k, l in mapping:
                val = self.extract_bool(els, l)
                if val is not None: section[k] = val
                else: section[k] = False # Default booleans in these sections to False if not checked

        d["construction"]["mainWalls"] = self.extract_text(els, ["Main Walls"])
        d["construction"]["mainRoof"] = self.extract_text(els, ["Main Roof"])
        d["construction"]["garageConstruction"] = self.extract_text(els, ["Garage:"])
        d["construction"]["outbuildingsConstruction"] = self.extract_text(els, ["Outbuildings:"])
        d["construction"]["alterationsAge"] = self.extract_int(els, ["years ago"])
        
        d["localityAndDemand"]["compulsoryPurchaseDetails"] = self.extract_text(els, ["Compulsory Purchase details"])
        d["localityAndDemand"]["vacantOrBoardedDetails"] = self.extract_text(els, ["vacant or boarded details"])
        d["localityAndDemand"]["occupancyRestrictionDetails"] = self.extract_text(els, ["Occupancy restriction details"])
        d["localityAndDemand"]["highVoltageEquipmentDetails"] = self.extract_text(els, ["high voltage equipment details"])
        
        d["services"]["centralHeatingType"] = self.extract_text(els, ["Central heating type"])
        d["services"]["easementsOrRightsDetails"] = self.extract_text(els, ["Easements details", "Rights of Way details"])

        # 7. Energy & Condition & Repairs
        d["energyEfficiency"]["epcRating"] = self.extract_text(els, ["EPC Rating"])
        d["energyEfficiency"]["epcScore"] = self.extract_int(els, ["EPC Score"])
        
        cop = d["conditionsOfProperty"]
        cop["isStructuralMovement"] = self.extract_bool(els, ["structural movement"])
        cop["isStructuralMovementHistoricOrNonProgressive"] = self.extract_bool(els, ["historic or non-progressive"])
        cop["structuralMovementDetails"] = self.extract_multiline(els, ["structural movement details"])
        cop["isStructuralModifications"] = self.extract_bool(els, ["structural modifications"])
        cop["structuralModificationsDetails"] = self.extract_multiline(els, ["structural modifications details"])
        cop["communalAreasMaintained"] = self.extract_bool(els, ["communal areas maintained"])
        for k in ["flooding", "subsidence", "heave", "landslip"]:
             cop["propertyProneTo"][k] = self.extract_bool(els, [k])
        cop["propertyProneTo"]["details"] = self.extract_text(els, ["prone to", "details"])
        cop["isPlotBoundariesDefinedUnderPointFourHectares"] = self.extract_bool(els, ["0.4 hectares"])
        cop["isTreesWithinInfluencingDistance"] = self.extract_bool(els, ["trees", "influencing distance"])
        cop["isBuiltOnSteepSlope"] = self.extract_bool(els, ["steep slope"])

        # 8. Reports & Rental & Valuation
        for k, l in [("isTimberDamp", ["Timber/Damp"]), ("isMining", ["Mining"]), ("isElectrical", ["Electrical"]), 
                     ("isDrains", ["Drains"]), ("isStructuralEngineers", ["Structural Engineer"]), ("isArboricultural", ["Arboricultural"]),
                     ("isMundic", ["Mundic"]), ("isWallTies", ["Wall Ties"]), ("isRoof", ["Roof"]), ("isMetalliferous", ["Metalliferous"])]:
            d["reports"][k] = self.extract_bool(els, l)
        
        ri = d["rentalInformation"]
        ri["isRentalDemandInLocality"] = self.extract_bool(els, ["rental demand"])
        ri["isOtherLettingDemandFactors"] = self.extract_bool(els, ["other letting demand factors"])
        ri["investorOnlyDemand"] = self.extract_bool(els, ["investor only demand"])
        ri["monthlyMarketRentPresentCondition"] = self.extract_currency(els, ["monthly market rent", "present"])
        ri["monthlyMarketRentImprovedCondition"] = self.extract_currency(els, ["monthly market rent", "improved"])
        
        vf = d["valuationForFinancePurpose"]
        vf["isSuitableForFinance"] = self.extract_bool(els, ["suitable security for finance"])
        vf["marketValuePresentCondition"] = self.extract_currency(els, ["present condition"])
        vf["marketValueAfterRepairs"] = self.extract_currency(els, ["after essential repairs"])
        vf["purchasePriceOrBorrowerEstimate"] = self.extract_currency(els, ["purchase price", "estimate"])
        vf["buildingInsuranceReinstatementCost"] = self.extract_currency(els, ["Reinstatement Cost"])
        
        d["generalRemarks"] = self.extract_multiline(els, ["GENERAL REMARKS"], max_lines=10)
        d["valuationForFinancePurposeHPP"] = copy.deepcopy(vf)
        
        vd = d["valuersDeclaration"]
        for k, l in [("mrics", ["MRICS"]), ("frics", ["FRICS"]), ("assocRics", ["AssocRICS"])]:
            vd["valuerQualifications"][k] = self.extract_bool(els, l)
        
        vd["valuerName"] = self.extract_text(els, ["Name of Valuer"])
        vd["onBehalfOf"] = self.extract_text(els, ["on behalf of"])
        vd["telephone"] = self.extract_int(els, ["Telephone"])
        vd["email"] = self.extract_text(els, ["E-mail"])
        vd["ricsNumber"] = self.extract_int(els, ["RICS Number"])
        vd["valuerAddress"] = self.extract_multiline(els, ["Address of Valuer"])
        vd["valuerPostcode"] = self.extract_text(els, ["Postcode"])
        vd["reportDate"] = self.extract_text(els, ["Report Date"])
        
        d["extractedText"] = " ".join([e["text"] for e in els])
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
