# Valuation Report Extraction System

A high-precision, coordinate-aware extraction engine designed to parse complex UK Valuation Report PDFs into structured JSON data aligned 100% with Mongoose schemas.

## 🚀 Overview

This system transforms unstructured, often poorly-OCR'd PDF valuation reports into machine-readable JSON. It uses a logic-gated approach to handle data fusion, misaligned tables, and complex Boolean markers that standard NLP parsers often miss.

## 🏗 Architecture & Workings

The system operates on a **Spatial-Proximity Logic** architecture:

1.  **OCR Layer**: Uses `PaddleOCR` to generate coordinate-bound text elements (BBBoxes).
2.  **Anchor Mapping**: Identifies "Anchor" keywords (labels) in the PDF using a multi-variant dictionary (handling OCR misreads like "Renu'r" for "Rent").
3.  **Proximity Search**:
    *   **Right-Search**: Looks for values horizontally aligned with high tolerance for merged text strings.
    *   **Bottom-Search**: Used for multiline comments and address blocks.
    *   **Spatial Boolean Parsing**: Analyzes fragments of strings (e.g., "Yes XNo") to determine precise check-box states.
4.  **Schema Alignment**: Post-processes data into a strictly typed structure defined in `schema_template.py`.

## 📊 Accuracy & Logic Features

*   **100% Schema Coverage**: Maps 160+ fields across 12+ nested sections.
*   **Data Fusion Resolution**: Robustly separates fused labels and values (e.g., "PostcodeLU7 1GN" → "LU7 1GN").
*   **Positional Disambiguation**: Correctly identifies "True" state even in merged Yes/No lines by analyzing character position.
*   **Boilerplate Suppression**: Filters out legal fine print and secondary labels to ensure data purity.
*   **Global Fallback**: Searches multiple pages for critical identifiers like Postcodes if primary slots are empty.

## 📂 Project Structure

*   `main.py`: The core extraction engine and logic controller.
*   `schema_template.py`: Python dictionary template aligned to the target Mongoose model.
*   `extracted_data.json`: The final structured output.
*   `raw_ocr_data.json`: A coordinate-aware cache of all detected PDF elements.
*   `poppler/`: Integrated binary layer for PDF-to-image conversion.

## 🛠 Setup & Running

### Prerequisites
*   Python 3.8+
*   PaddleOCR & PaddlePaddle
*   Poppler (included in project directory)

### Installation
```bash
# Initialize virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies (ensure paddlepaddle-gpu or paddlepaddle is installed)
pip install paddleocr paddlepaddle pdf2image opencv-python numpy
```

### Execution
Simply run the main script to process the test files:
```bash
python main.py
```

## 📥 Inputs & 📤 Outputs

*   **Input**: PDF files (e.g., `Ashok.test Valuation Report.pdf`, `All yes fields (1).pdf`).
*   **Output**:
    *   `extracted_data.json`: Key-value pairs organized by sections (Property Type, Accommodation, etc.).
    *   Example:
        ```json
        "tenure": "Freehold",
        "groundRent": 1717.0,
        "isUrban": true,
        "postCode": "LU7 1GN"
        ```

## 📈 Current Status
*   **Schema Alignment**: 100% complete.
*   **Refractive Accuracy**: High. Handles misreads, column drifts, and multi-page reports.
*   **Test Coverage**: Validated against "Standard", "All Fields Check", and "Empty" report variants.